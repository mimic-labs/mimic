import torch
import torch.nn as nn
import torch.optim as optim
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TGT_LEN = 5 # TODO: don't hardcode this (set to num_captures for now)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # create a long enough `pe` matrix that can be truncated to the actual sequence length in forward pass
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # make `pe` a persistent buffer to avoid it being considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # add positional encoding to each embedding in input sequence
        x = x + self.pe[:x.size(0), :]
        return x
    
class TransformerSeq2Seq(nn.Module):
    def __init__(self, object_dim, hand_dim, hidden_dim, nhead, num_layers, max_seq_len=5000):
        super().__init__()
        self.object_encoder = nn.Linear(object_dim, hidden_dim)
        self.hand_encoder = nn.Linear(hand_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        # self.output_decoder = nn.Linear(hidden_dim, object_dim + hand_dim)
        self.output_decoder = nn.Linear(hidden_dim, hand_dim)
    
    def get_mask(self, sz, context_len=0):
        mask = nn.Transformer.generate_square_subsequent_mask(sz).to(device)
        mask[:context_len, :context_len] = 0 # unmask the prepended context vector
        
        return mask
    
    def forward(
        self,
        ref_obj_embeddings,
        ref_obj_pos,
        ref_hand_pos,
        new_obj_embeddings,
        new_obj_pos,
        tgt_hand_pos,
    ):
        ref_object_data = torch.cat([ref_obj_embeddings, ref_obj_pos], dim=-1)
        ref_object_data = self.object_encoder(ref_object_data).flatten(start_dim=2)
        print(ref_object_data.shape)
        
        ref_hand_data = self.hand_encoder(ref_hand_pos).flatten(start_dim=2)
        print(ref_hand_pos.shape)
        # combined_ref_data = torch.cat((ref_object_data, ref_hand_data.unsqueeze(1)), dim=1)
        combined_ref_data = torch.cat((ref_object_data, ref_hand_data), dim=1)
        print(combined_ref_data.shape)
        
        combined_ref_data = self.positional_encoding(combined_ref_data)
        memory = self.transformer_encoder(combined_ref_data)
        
        print(memory, memory.shape)
        
        new_object_data = torch.cat([new_obj_embeddings, new_obj_pos], dim=-1)
        new_object_data = self.object_encoder(new_object_data).flatten(start_dim=2)
        
        tgt_hand_pos = self.hand_encoder(tgt_hand_pos).flatten(start_dim=2)
        
        combined_tgt_data = torch.cat((new_object_data, tgt_hand_pos), dim=1)
        combined_tgt_data = self.positional_encoding(combined_tgt_data)

        tgt_mask = self.get_mask(len(combined_tgt_data), context_len=len(new_object_data))
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(combined_tgt_data)).to(ref_obj_embeddings.device)
        # tgt_mask[:len(new_object_data), :len(new_object_data)] = 0  # Allow decoder to see only the new object data
        
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
        new_obj_pos
    ):
        ''' src has dimension of LEN x 1 '''
        ref_object_data = torch.cat([ref_obj_embeddings, ref_obj_pos], dim=-1)
        ref_object_data = self.object_encoder(ref_object_data).flatten(start_dim=2)
        print(ref_hand_pos.shape)
        
        ref_hand_data = self.hand_encoder(ref_hand_pos).flatten(start_dim=2)
        combined_ref_data = torch.cat((ref_object_data, ref_hand_data), dim=1)
        print(combined_ref_data.shape)
        
        combined_ref_data = self.positional_encoding(combined_ref_data)
        memory = self.transformer_encoder(combined_ref_data)
        
        new_object_data = torch.cat([new_obj_embeddings, new_obj_pos], dim=-1)
        new_object_data = self.object_encoder(new_object_data).flatten(start_dim=2)
        
        SOS_TOKEN = torch.LongTensor([0]).view(-1,1).to(device)
        
        prefix = new_object_data#[*new_object_data]
        inputs = [SOS_TOKEN]
        
        for i in range(MAX_TGT_LEN):
            tgt = torch.LongTensor([inputs]).view(-1,1).to(device)
            
            tgt = self.hand_encoder(tgt)
            
            combined_tgt = torch.cat((prefix, tgt), dim=1)
            
            combined_tgt = self.positional_encoding(tgt)
            
            combined_tgt_mask = self.get_mask(i+1, context_len=len(prefix)).to(device)
            
            output = self.transformer_decoder(
                tgt=combined_tgt, 
                memory=memory, 
                tgt_mask=combined_tgt_mask,
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