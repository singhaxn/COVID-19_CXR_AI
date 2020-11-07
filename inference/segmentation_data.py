import os
import numpy as np
from scipy.ndimage import label, find_objects, generate_binary_structure

# >> For training only

dataset_dir = "/data/datasets/NLM-shenzhen-montgomery/original+pediatric"
train_path = os.path.join(dataset_dir, "train")

def train_mask_fn(x):
    return os.path.join(train_path, "mask", f"{x.stem}.png")

# << For training only


label_structure = generate_binary_structure(2,2)

def find_bounds(mask, exclude_threshold=0.0, padding=0.0):
    labeled_array, num_features = label(mask, label_structure)
    objects = find_objects(labeled_array)
    
    bounds = [mask.shape[1], mask.shape[0], 0, 0]
    
    for o in objects:
        area = (o[0].stop - o[0].start) * (o[1].stop - o[1].start)
        if area > mask.shape[0] * mask.shape[1] * exclude_threshold:
            if o[1].start < bounds[0]:
                bounds[0] = o[1].start
            if o[0].start < bounds[1]:
                bounds[1] = o[0].start
            if o[1].stop > bounds[2]:
                bounds[2] = o[1].stop
            if o[0].stop > bounds[3]:
                bounds[3] = o[0].stop
    
#     print(bounds)
#     pady, padx = np.array(mask.shape) * padding
    padx = int((bounds[2] - bounds[0]) * padding)
    pady = int((bounds[3] - bounds[1]) * padding)
    bounds[0] = max(0, bounds[0] - padx)
    bounds[1] = max(0, bounds[1] - pady)
    bounds[2] = min(mask.shape[1], bounds[2] + padx)
    bounds[3] = min(mask.shape[0], bounds[3] + pady)
#     print(bounds)
    
    return bounds

def compute_confidence_score(pred, mask):
    mask_sum = mask.sum()
    if mask_sum == 0:
        return 0.
    else:
        mask_preds = pred[1] * mask
    #     confidence = np.true_divide(mask_preds.sum(), (mask_preds!=0).sum())
        return np.true_divide(mask_preds.sum(), mask_sum).numpy()
