# Importing Numpy, Pandas, Matplotlib and Torch related modules
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json

# Importing modules from pred_utils (utilities)
import pred_utils
from pred_utils import parsing
from pred_utils import load_checkpoint
from pred_utils import process_image

# Making predictions by loading a trained network
if __name__ == '__main__':
    arguments = parsing()
    
    image = process_image(arguments.image_directory)
    image.unsqueeze_(0)
    
    # Switching to device (cpu/gpu) chosen by the user
    if arguments.gpu == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    # Loading model from a checkpoint
    print(device)
    model = load_checkpoint(arguments.checkpoint, device).cpu()
    model.eval()
    image.to(device)
 
    # Calculate top k probabilities and associated class 
    with torch.no_grad():
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(int(arguments.top_k), dim=1)
    
    probabilities_topk = np.array(top_p)[0]
    indexes_topk = np.array(top_class)[0]
    classes_to_index = model.class_to_idx
    
    # Inverting the dictionary saved in the mopel which maps indices to classes
    index_to_classes = {x: y for y, x in classes_to_index.items()}
    
    classes_topk = []
    for index in indexes_topk:
        classes_topk += [index_to_classes[index]]
    
    names = []
    with open(arguments.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    for i in classes_topk:
        names += [cat_to_name[i]]
    
    # Printing output
    print(probabilities_topk)
    print(names)
