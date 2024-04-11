import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, embeddings, ref_object_positions, ref_hand_pos, new_object_embeddings, new_object_positions, correct_hand_pos):
        self.embeddings = embeddings
        self.ref_object_positions = ref_object_positions
        self.ref_hand_pos = ref_hand_pos
        self.new_object_embeddings = new_object_embeddings
        self.new_object_positions = new_object_positions
        self.correct_hand_pos = correct_hand_pos
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embeddings': self.embeddings[idx],
            'ref_object_positions': self.ref_object_positions[idx],
            'ref_hand_pos': self.ref_hand_pos[idx],
            'new_object_embeddings': self.new_object_embeddings[0],  # assuming always the same for all samples
            'new_object_positions': self.new_object_positions[0],  # assuming always the same for all samples
            'correct_hand_pos': self.correct_hand_pos[idx]
        }
    
    # def __str__(self):
    #     return f"CustomDataset with {len(self)} samples"
        
if __name__ == "__main__":
    num_objects = 10
    num_captures = 5
    
    dataset = CustomDataset(
        embeddings=torch.randn(num_captures, num_objects, 128),  # Example data
        ref_object_positions=torch.randn(num_captures, num_objects, 12),
        ref_hand_pos=torch.randn(num_captures, 3),
        new_object_embeddings=torch.randn(1, num_objects, 128),
        new_object_positions=torch.randn(1, num_objects, 12),
        correct_hand_pos=torch.randn(num_captures, 3)
    )