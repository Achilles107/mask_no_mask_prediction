import json
import torch

from commons import get_model, get_tensor

with open('category.json') as f:
    cat_to_name = json.load(f)

with open('cat_to_idx.json') as f:
    cat_to_idx = json.load(f)

idx_to_cat = {v:k for k, v in cat_to_idx.items()}
model = get_model()

def get_flower_name(image_bytes):
    tensor = get_tensor(image_bytes=image_bytes)
    outputs = model(tensor).view(-1)
    preds = torch.sigmoid(outputs)
    preds = torch.round(preds)
    class_idx = idx_to_cat[int(preds.item())]
    mask_or_not = cat_to_name[class_idx]
    print(mask_or_not)
    return mask_or_not