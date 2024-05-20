import torch
import torch.nn as nn
import torch.optim as optim
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TGT_LEN = 5 # TODO: don't hardcode this (set to num_captures for now)

class AttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionAggregator, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, num_objects+1, input_dim]
        q = self.query(x)  # [batch_size, seq_len, num_objects+1, output_dim]
        k = self.key(x)    # [batch_size, seq_len, num_objects+1, output_dim]
        v = self.value(x)  # [batch_size, seq_len, num_objects+1, output_dim]

        # Attention score calculation
        scores = torch.einsum('bsiq,bsjq->bsij', q, k)  # [batch_size, seq_len, num_objects+1, num_objects+1]
        weights = torch.softmax(scores, dim=-1)  # Softmax over the last dimension (num_objects+1)

        # Weighted sum of values
        x = torch.einsum('bsij,bsjd->bsid', weights, v)  # [batch_size, seq_len, num_objects+1, output_dim]
        # x = x.sum(dim=2)  # Sum over num_objects+1 dimension, result is [batch_size, seq_len, output_dim]
        x = x.mean(dim=2)
        return x

# batch first
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.emb_size = emb_size  # Store the embedding size for checks
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        # Reshape for batch first [1, maxlen, emb_size]
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        """
        Args:
        token_embedding : Tensor, shape [batch_size, seq_length, embedding_dim]
        """
        if token_embedding.size(2) != self.emb_size:
            raise ValueError(f"Token embeddings dimension {token_embedding.size(2)} must match "
                             f"positional embeddings dimension {self.emb_size}")

        # apply positional encoding with broadcasting
        # self.pos_embedding is [1, maxlen, emb_size] and automatically adjusts to the batch size
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])
    
class TransformerSeq2Seq(nn.Module):
    def __init__(self, object_dim, hand_dim, hidden_dim, nhead, num_layers, batch_first = True, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.object_encoder = nn.Linear(object_dim, hidden_dim)
        self.hand_encoder = nn.Linear(hand_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout, max_seq_len)
        self.attn_agg = AttentionAggregator(hidden_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=batch_first)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        # self.output_decoder = nn.Linear(hidden_dim, object_dim + hand_dim)
        self.output_decoder = nn.Linear(hidden_dim, hand_dim)
    
    def get_mask(self, sz, context_len=0):
        mask = nn.Transformer.generate_square_subsequent_mask(sz).to(device)
        mask[:context_len, :context_len] = 0 # unmask the prepended context vector
        
        return mask
    
    def forward(
        self,
        ref_obj_embeddings: torch.Tensor,
        ref_obj_pos: torch.Tensor,
        ref_hand_pos: torch.Tensor,
        new_obj_embeddings: torch.Tensor,
        new_obj_pos: torch.Tensor,
        tgt_hand_pos: torch.Tensor,
    ):
        ref_object_data = torch.cat([ref_obj_embeddings, ref_obj_pos], dim=-1)
        ref_object_data = self.object_encoder(ref_object_data)#.flatten(start_dim=2)
        # print(f"{ref_object_data.shape=}")
        
        ref_hand_data = self.hand_encoder(ref_hand_pos)#.flatten(start_dim=2)
        # print(f"{ref_hand_pos.shape=}")
        # combined_ref_data = torch.cat((ref_object_data, ref_hand_data.unsqueeze(1)), dim=1)
        combined_ref_data = torch.cat((ref_object_data, ref_hand_data), dim=2)
        # print(f"{/combined_ref_data.shape=}")
        
        combined_ref_data = self.attn_agg(combined_ref_data)
        combined_ref_data = self.positional_encoding(combined_ref_data)
        
        
        # print(f"{combined_ref_data.shape=}")
        
        
        memory = self.transformer_encoder(combined_ref_data)
        
        # print(memory, memory.shape)
        
        new_object_data = torch.cat([new_obj_embeddings, new_obj_pos], dim=-1)
        new_object_data = self.object_encoder(new_object_data)#.flatten(start_dim=2)
        
        tgt_hand_data = self.hand_encoder(tgt_hand_pos)#.flatten(start_dim=2)
        
        combined_tgt_data = torch.cat((new_object_data, tgt_hand_data), dim=2)
        combined_tgt_data = self.attn_agg(combined_tgt_data)
        combined_tgt_data = self.positional_encoding(combined_tgt_data)
        
        

        # print(f"{combined_tgt_data.shape=}")

        tgt_mask = self.get_mask(combined_tgt_data.size(1))
        # tgt_mask = self.get_mask(len(combined_tgt_data), context_len=len(new_object_data))
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(combined_tgt_data)).to(ref_obj_embeddings.device)
        # tgt_mask[:len(new_object_data), :len(new_object_data)] = 0  # Allow decoder to see only the new object data
        # print(tgt_mask.shape, combined_tgt_data.shape, memory.shape)
        output_sequence = self.transformer_decoder(
            tgt = combined_tgt_data, 
            memory = memory, 
            tgt_mask = tgt_mask, # to avoid looking at the future tokens (the ones on the right)
            # tgt_key_padding_mask = tgt_key_padding_mask, # to avoid working on padding
            # memory_key_padding_mask = src_key_padding_mask # avoid looking on padding of the src
        )
        
        output_sequence = self.output_decoder(output_sequence) # may need to remove sequence dim?
        
        return output_sequence
    
    def generate(
        self,
        ref_obj_embeddings,
        ref_obj_pos,
        ref_hand_pos,
        new_obj_embeddings,
        new_obj_pos,
        start_hand_pos=None
    ):
        ''' src has dimension of LEN x 1 '''
        ref_object_data = torch.cat([ref_obj_embeddings, ref_obj_pos], dim=-1)
        ref_object_data = self.object_encoder(ref_object_data)#.flatten(start_dim=2)
        print(ref_hand_pos.shape)
        
        ref_hand_data = self.hand_encoder(ref_hand_pos)#.flatten(start_dim=2)
        combined_ref_data = torch.cat((ref_object_data, ref_hand_data), dim=1)
        print(combined_ref_data.shape)
        
        combined_ref_data = self.positional_encoding(combined_ref_data)
        memory = self.transformer_encoder(combined_ref_data)
        
        new_object_data = torch.cat([new_obj_embeddings, new_obj_pos], dim=-1)
        new_object_data = self.object_encoder(new_object_data)#.flatten(start_dim=2)
        
        # SOS_TOKEN = torch.LongTensor([0]).view(-1,1).to(device)
        start_hand_pos = torch.zeros(1,1,3).to(device) if start_hand_pos is None else start_hand_pos
        start_hand_data = self.hand_encoder(start_hand_pos)#.flatten(start_dim=2)
        
        START = torch.cat((new_object_data, start_hand_data), dim=1)
        # prefix = new_object_data#[*new_object_data]
        inputs = [START]
        
        for i in range(MAX_TGT_LEN):
            tgt = torch.LongTensor([inputs]).view(-1,1).to(device)
            tgt_mask = self.get_mask(i+1).to(device)
            
            tgt = self.positional_encoding(tgt)
            output = self.transformer_decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                # memory_key_padding_mask = src_key_padding_mask
            )
            
            output = self.output_decoder(output)
            print(output, output.shape)
            # output = self.linear(output)
            # output = self.softmax(output)
            # output = output[-1] # the last timestep
            # values, indices = output.max(dim=-1)
            pred_token = output[-1]
            print(pred_token, pred_token.shape)
            inputs.append(pred_token)

        return inputs[1:]