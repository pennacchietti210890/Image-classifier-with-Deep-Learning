# utilities to be used in train.py
import argparse
import torch
from torchvision import datasets, transforms, models

def parsing():
    # Parsing arguments from the command line
    parser = argparse.ArgumentParser(description = 'Parsing command line options for training the network')
    # Required arguments: data directory (This is assumed to be the directory containing training, validation and testing data)
    parser.add_argument('data_directory',  nargs='?', action = 'store', default = 'flowers')
    # Optional arguments: save directory / architecture / learning rate / hidden units / epochs
    parser.add_argument('--save_dir', action = 'store', dest = 'save_directory', default = '')
    parser.add_argument('--arch', action = 'store', dest = 'architecture', default = models.vgg11(pretrained = True))
    parser.add_argument('--learning_rate', action = 'store', dest = 'learn_rate', default = 0.001)
    parser.add_argument('--hidden_units', action = 'store', dest = 'h_units', default = 200)
    parser.add_argument('--output_units', action ='store', dest = 'out_units', default = 102)
    parser.add_argument('--epochs', action = 'store', dest = 'epochs', default = 5)
    parser.add_argument('--gpu', action = 'store_true', default = False)
    args = parser.parse_args()
    return args

def transforming(path):
    # Transform image data into tensors to be used in PyTorch 
    
    # Getting the specific paths to each of the training, validation and testing folder
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Preparing the transformations needed to convert images into tensors 
    data_transforms = {'training': transforms.Compose([transforms.RandomRotation(30), transforms.RandomHorizontalFlip(30),
                                                  transforms.RandomResizedCrop(224), transforms.ToTensor(), 
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])]), 
                   
                   
                   'validation' : transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), 
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])]), 

    
                   'testing': transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), 
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])])}

    # Loading the data from the directories
    image_datasets = {'training': datasets.ImageFolder(train_dir, transform = data_transforms['training']),       
                     'validation': datasets.ImageFolder(valid_dir, transform = data_transforms['validation']),
                     'testing':datasets.ImageFolder(test_dir, transform = data_transforms['testing'])}

    # Batch iterators to feed the network
    dataloaders = {'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size = 100, shuffle = True),
                   'validation':torch.utils.data.DataLoader(image_datasets['validation'], batch_size = 100, shuffle = True),
                    'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size = 100, shuffle = True)}
    
    
    return dataloaders, image_datasets

    




