import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import TransformerSeq2Seq
from model.simulate_data import generate_train_data, visualize_captures

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
num_examples = 500

dataset = generate_train_data(num_captures, num_objects, num_examples)

# DataLoader for batching
data_loader = DataLoader(dataset, batch_size=5, shuffle=True)


# initialize model
model = TransformerSeq2Seq(object_dim, hand_dim, hidden_dim, nhead, num_layers).to(device)

# model training setup
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        # for x in data:
        #     print(x, data[x].dtype, data[x].shape)
        
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
        # print(f"Predicted hand pos: {predicted_hand_pos}; {predicted_hand_pos.shape}\nCorrect hand pos: {correct_hand_pos}; {correct_hand_pos.shape}")
        
        # calculate Loss
        loss = loss_function(predicted_hand_pos.unsqueeze(2), correct_hand_pos)
        
        # backward pass + optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # print every 10 mini-batches
            print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 10}')
            running_loss = 0.0

print('Training complete')

num_test = 5

val_data_all = generate_train_data(num_captures, num_objects, num_test)

val_loader = DataLoader(val_data_all, batch_size=1, shuffle=True)
print(len(val_loader))
for i,val_data in enumerate(val_loader, 0):
    predicted_hand_pos = model(
        ref_obj_embeddings=val_data['embeddings'].to(device),
        ref_obj_pos=val_data['ref_object_positions'].to(device),
        ref_hand_pos=val_data['ref_hand_pos'].to(device),
        new_obj_embeddings=val_data['new_object_embeddings'].to(device),
        new_obj_pos=val_data['new_object_positions'].to(device),
        tgt_hand_pos=val_data['correct_hand_pos'].to(device)
    )
    print(f"{predicted_hand_pos.shape=}")
    print(f"{predicted_hand_pos.squeeze(0).unsqueeze(1).shape=}")
    xy_pred = predicted_hand_pos.squeeze(0).unsqueeze(1).cpu().detach().numpy()[
        :, 0, :2
    ]
    print(f"{xy_pred.shape=}")
    print(xy_pred)        
    visualize_captures(
        num_captures,
        num_objects,
        val_data['new_object_embeddings'].squeeze(),
        val_data['new_object_positions'].squeeze(),
        true_hand_pos=val_data['correct_hand_pos'].squeeze(0),
        pred_hand_pos=predicted_hand_pos.squeeze(0).unsqueeze(1)
    )