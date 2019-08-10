# Importing Numpy, Pandas, Matplotlib and Torch related modules
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Importing modules from train_utils (utilities)
import train_utils
from train_utils import parsing
from train_utils import transforming

# Training the network
if __name__ == '__main__':
    arguments = parsing()
    
    # Loading the data
    dataloaders = transforming(arguments.data_directory)[0]
    datasets = transforming(arguments.data_directory)[1]
    
    # Switching to device (cpu/gpu) chosen by the user
    if arguments.gpu == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    # Loading the pre-trained model chosen by the user 
    model = arguments.architecture
    n_features = model.classifier.__getitem__(0).in_features
 
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(n_features, int(arguments.h_units)),
                            nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(int(arguments.h_units), int(arguments.out_units)),
                           nn.LogSoftmax(dim = 1))

    criterion = nn.NLLLoss()

    # Set optimizer with learning rate as specified by the user
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(arguments.learn_rate))

    # Switching model to whatever device chosen by the user
    model.to(device)

    # Training and Validating the model
    epochs = int(arguments.epochs)
    print_every = 20
    
    for epoch in range(epochs):    
        train_loss = 0
        val_loss = 0
        accuracy = 0
        for inputs, labels in dataloaders['training']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
       
       # Evaluating the model on the validation set at each epoch
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['validation']:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                val_loss+= loss.item()
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        acc = accuracy / len(dataloaders['validation'])
        
        # Printing output
        print(f"Epoch {epoch+1}/{epochs}.. " f"training set loss: {train_loss:.3f}..")
        print(f"Epoch {epoch+1}/{epochs}.. " f"Validation set loss: {val_loss:.3f}..")
        print(f"Epoch {epoch+1}/{epochs}.. " f"Validation set accuracy: {acc:.3f}..")

    model.train()
                
    # Saving a checkpoint once the model has been trained
    checkpoint = {'state_dict': model.state_dict(),
                    'epochs': epochs,
                    'classifier': model.classifier,
                    'class_to_idx': datasets['training'].class_to_idx}

    path = '/home/workspace/ImageClassifier/' + arguments.save_directory + '/checkpoint.pth'
    print("Saving the model to: " + path)
    torch.save(checkpoint, path)
                
        

            












