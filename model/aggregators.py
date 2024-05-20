import torch
import torch.nn as nn

class Conv1DAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Conv1DAggregator, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, num_objects+1, input_dim]
        x = x.permute(0, 1, 3, 2)  # Change to [batch_size, seq_len, input_dim, num_objects+1]
        x = self.conv(x)  # Apply convolution across the num_objects+1 dimension
        x = x.squeeze(-1)  # Reduce the last dimension
        return x

# Example dimensions
batch_size = 10
seq_length = 50
num_objects = 10  # Can vary
input_dim = 64
output_dim = 64  # Desired embedding size

# Input tensor
src = torch.rand(batch_size, seq_length, num_objects+1, input_dim)

# Model
# aggregator = Conv1DAggregator(input_dim, output_dim)
# output = aggregator(src)
# print(output.shape)  # Expected shape: [batch_size, seq_length, output_dim]

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
        x = x.sum(dim=2)  # Sum over num_objects+1 dimension, result is [batch_size, seq_len, output_dim]
        return x

# Model
attention_aggregator = AttentionAggregator(input_dim, output_dim)
output = attention_aggregator(src)
print(output.shape)  # Expected shape: [batch_size, seq_length, output_dim]
