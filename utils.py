import os
import torch

realmin = 1e-6


def norm(input, p=2, dim=-1, eps=1e-12):
    #return torch.linalg.norm(input,p ,dim, keepdim=True).clamp(min=eps).expand_as(input)
    return torch.linalg.norm(input,p ,dim, keepdim=True).clamp(min=eps)

def return_norm(input, p=2, dim=-1, eps=1e-12):
    # return torch.linalg.norm(input,p ,dim, keepdim=True).clamp(min=eps).expand_as(input)
    return input/torch.linalg.norm(input, p, dim, keepdim=True).clamp(min=eps)
#return input.pow(2).sum(dim=dim).sqrt().unsqueeze(-1)

def mkdirs(path):    
    if not os.path.exists(path):
        os.makedirs(path)

