# utilities to be used in predict.py
import argparse
import torch
from torchvision import datasets, transforms, models
from PIL import Image

def parsing():
    # Parsing arguments from the command line
    parser = argparse.ArgumentParser(description = 'Parsing command line options for training the network')
    # Required arguments: data directory (This is assumed to be the file path of the image) / checkpoint
    parser.add_argument('image_directory', nargs='?', action = 'store', default = 'flowers/test/1/image_06743.jpg')
    parser.add_argument('checkpoint',  nargs='?', action = 'store', default = 'checkpoint.pth')
    # Optional arguments: top K most likely classes / mapping categories file / gpu
    parser.add_argument('--top_k', action = 'store', default = 5)
    parser.add_argument('--category_names', action = 'store', default = 'cat_to_name.json')
    parser.add_argument('--gpu', action = 'store_true', default = False)
    args = parser.parse_args()
    return args

def load_checkpoint(filepath, device, pretrained_model = models.vgg11(pretrained=True)):
    # loading a model from a checkpoint file
    checkpoint = torch.load(filepath)
    # Reconstruct the model starting from a pre-trained model and adding a classifier architecture
    model.classifier = checkpoint['classifier']
    # Loading the state dict
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    image_transformed = transform(pil_image)
    return image_transformed
                                   