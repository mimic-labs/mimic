import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import TransformerSeq2Seq
from model.simulate_data import generate_train_data

from torch.utils.data import Dataset, DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_objects = 10
object_dim = 140 #(10*140) # 10*(128 embeddings + 12 positions)
hand_dim = 3
hidden_dim = 256
nhead = 8
num_layers = 6

num_captures = 5
num_examples = 10

dataset = generate_train_data(num_captures, num_objects, num_examples)

# DataLoader for batching
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)


# initialize model
model = TransformerSeq2Seq(object_dim, hand_dim, hidden_dim, nhead, num_layers).to(device)

# model training setup
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        for x in data:
            print(x, data[x].dtype, data[x].shape)
        
        # unpack data
        embeddings = data['embeddings'].to(device)#.flatten(start_dim=2)
        ref_object_positions = data['ref_object_positions'].to(device)#.flatten(start_dim=2)
        ref_hand_pos = data['ref_hand_pos'].to(device)#.flatten(start_dim=2)
        new_object_embeddings = data['new_object_embeddings'].to(device)#.flatten(start_dim=2)
        new_object_positions = data['new_object_positions'].to(device)#.flatten(start_dim=2)
        correct_hand_pos = data['correct_hand_pos'].to(device)#.flatten(start_dim=2)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        predicted_hand_pos = model(
            ref_obj_embeddings=embeddings,
            ref_obj_pos=ref_object_positions,
            ref_hand_pos=ref_hand_pos,
            new_obj_embeddings=new_object_embeddings,
            new_obj_pos=new_object_positions,
            tgt_hand_pos=correct_hand_pos
            # new_hand_pos,
            # tgt_seq_length=len(correct_hand_pos)  # Assuming target sequence length is the same as the number of captures
        )
        print(f"Predicted hand pos: {predicted_hand_pos}; {predicted_hand_pos.shape}\nCorrect hand pos: {correct_hand_pos}; {correct_hand_pos.shape}")
        
        # calculate Loss
        loss = loss_function(predicted_hand_pos, correct_hand_pos)
        
        # backward pass + optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # print every 10 mini-batches
            print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 10}')
            running_loss = 0.0

print('Training complete')
