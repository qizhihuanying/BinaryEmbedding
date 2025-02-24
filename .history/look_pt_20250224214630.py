import torch

file_path = "./project/models/binary_head/binary_head_full.pt"
state_dict = torch.load(file_path, map_location="cpu")

for key, value in state_dict.items():
    print(f"{key}: {value}")
