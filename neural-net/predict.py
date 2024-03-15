import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Define the transformer model
class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super(TransformerSeq2Seq, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout),
            num_layers)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(input_dim, num_heads, hidden_dim, dropout),
            num_layers)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, src, tgt):
        src = src.permute(1, 0, 2)  # Change input shape to (seq_len, batch_size, input_dim)
        tgt = tgt.permute(1, 0, 2)  # Change target shape to (seq_len, batch_size, input_dim)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.linear(output)
        return output

# Define a custom dataset class
class MatrixXYZDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Dummy data (replace with your actual data)
class MatrixXYZSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# Example usage
input_dim = 100  # Dimensionality of input matrices
output_dim = 3  # Dimensionality of output XYZ locations
hidden_dim = 256  # Hidden dimension of transformer layers
num_layers = 4  # Number of transformer layers
num_heads = 8  # Number of attention heads
dropout = 0.1  # Dropout probability

# Initialize transformer model
model = TransformerSeq2Seq(input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy input and target data
input_data = [torch.randn(3, input_dim),  # Matrix with shape (3, input_dim)
              torch.randn(4, input_dim),  # Matrix with shape (4, input_dim)
              torch.randn(5, input_dim)]  # Matrix with shape (5, input_dim)
target_data = [torch.tensor([1.0, 2.0, 3.0]),  # XYZ location for first sequence
               torch.tensor([4.0, 5.0, 6.0]),  # XYZ location for second sequence
               torch.tensor([7.0, 8.0, 9.0])]  # XYZ location for third sequence

# Convert input_data to padded batch format
padded_input_data = pad_sequence(input_data, batch_first=True, padding_value=0.0)

# Convert target_data to padded batch format
padded_target_data = pad_sequence(target_data, batch_first=True, padding_value=0.0)

# Create datasets
train_dataset = MatrixXYZSequenceDataset(list(zip(padded_input_data, padded_target_data)))

# Create data loader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        optimizer.zero_grad()
        output_seq = model(input_seq, target_seq[:, :-1, :])  # Predict XYZ locations, excluding last target
        loss = criterion(output_seq, target_seq[:, 1:, :])  # Compare predicted XYZ locations with true target
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}')
