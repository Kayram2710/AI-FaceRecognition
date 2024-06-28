#This script will train our main model and save it in its entierty
import torch
import torch.nn as network
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from ModelsScript.ModelMain import FaceRecognitionModel as Main
from ModelsScript.ModelVarient1 import FaceRecognitionModel as V1
from ModelsScript.ModelVarient2 import FaceRecognitionModel as V2

#Define function for returning dataset
def getDataset(path = "Cleaned Dataset"):

    #Defining transform operations
    transform = transforms.Compose([
        transforms.Grayscale(1), #Grayscale
        transforms.ToTensor(), #Turn data into readable tensors          
        transforms.Normalize((0.5,), (0.5,)) #Normalize data
    ])

    #Create variable to existing dataset based on an image folder
    dataset = datasets.ImageFolder(   
        root=path,              #param1: path to dataset
        transform=transform     #param2: point to transform operation
    )

    #Splitting the data, whats currently a blank variable will later be used for testing 
    remainder, _ = train_test_split(
        np.arange(len(dataset.targets)),    #param1: list of all indices in the dataset
        test_size=0.15,                     #param2: portion of split data (15%)
        random_state=50,                    #param3: seed for shuffling
        shuffle=True,                       #param4: enable shuffling
        stratify=dataset.targets            #param5: strasfying in order to have proportionate types of labels
    )

    #Splitting the data between the training set and validation set
    remainder, validationIndices = train_test_split(
        remainder,                                          #param1: list of all indices in the remaining dataset
        test_size=0.1765,                                   #param2: portion of split data (15% of remaining 85%)
        random_state=50,                                    #param3: seed for shuffling
        shuffle=True,                                       #param4: enable shuffling
        stratify=[dataset.targets[i] for i in remainder]    #param5: strasfying in order to have proportionate types of labels
    )

    #Creating the datasets based off the indices generated 
    trainingData = Subset(dataset, remainder) #The remainders serve as the training data
    validationData = Subset(dataset, validationIndices) #The data from the second split serves as a validation group

    #Create operationable dataset
    trainingData = DataLoader(  #Call data loader to load training data
        dataset=trainingData,   #param1: dataset points to a newly created dataset "training data"          
        batch_size=40,          #param2: limit batch size to 40
        shuffle=True            #param3: set shuffle to True
    )

    #Create operationable dataset
    validationData = DataLoader(    #Call data loader to load validation data
        dataset=validationData,     #param1: dataset points to a newly created dataset "validationData"          
        batch_size=40,              #param2: limit batch size to 40
        shuffle=True                #param3: set shuffle to True
    )

    #return created dataset
    return trainingData, validationData

#Define function to train the model
def train(path,  type=0, epochs = 15, treshold = 2):

    #Create device to run model
    #This is to allow the the program to run on a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Writing parameters for early stoping procedure
    fluctuationTreshold = treshold #Patience level set to two by default
    countFluctuation = 0 #Count start at 0
    PreviousLoss = 0 #Create variable to store previous loss

    #Instantiate Model Depending on type fed
    if(type == 1):
        model = V1().to(device)
    elif(type == 2):
        model = V2().to(device)
    else:
        model = Main().to(device) #Allows toss up path

    #Create a call for the "Cross Entrop Loss" function
    #This function will calculate the differance across 
    criterion = network.CrossEntropyLoss()

    #Setting an Adam optimizer
    #This will allow the model to adjust itself based on the returned loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Retrieve Dataset
    dataset, valDataset = getDataset()

    #Start looping through epochs
    for epoch in range(epochs):

        #Setting model to training mode
        model.train()

        #For loop through each image in the dataset and return the images and their label
        for images, labels in dataset:

            #Moving tensors to the chosen device
            images = images.to(device)
            labels = labels.to(device)

            #retrive the model's prediction by passing images through label
            modelPredictions = model(images)

            #Compute the differance between the predicted labels and actual labels by using the "Cross Entrop Loss" through criterion
            #This is refred to as "loss"
            loss = criterion(modelPredictions, labels)

            #Clear gradient(rate of differance) from the optimizer to allow for space during the next epochs
            optimizer.zero_grad()

            #Performs back propagation, computing change in loss since last step
            loss.backward()

            #Optimize accordingly
            optimizer.step()

        #Setting model to validation phase
        model.eval()
     
        #For loop through each image in the validation dataset and return the images and their label
        for images, labels in valDataset:

            #Moving tensors to the chosen device
            images = images.to(device)
            labels = labels.to(device)

            #retrive the model's prediction by passing images through label
            validationPrediction = model(images)

            #Compute the differance between the predicted labels and actual labels by using the "Cross Entrop Loss" through criterion
            validationLoss = criterion(validationPrediction, labels)
        
        #Output result
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {validationLoss.item():.4f}')

        #Early stop procedure
        #If epoch is at 0, do nothing
        if(epoch == 0):
            ""
        #Else check if new loss is greater then previous
        elif(validationLoss.item() >= PreviousLoss):
            countFluctuation = 1 + countFluctuation #If it is count the fluctuation
        else:
            countFluctuation = 0 #If it isnt reset it
        
        #Note down previous loss
        PreviousLoss = validationLoss.item()

        #Then check if count equal or surpasses treshold
        if(countFluctuation >= fluctuationTreshold):
            print(f'End of training due to too many fluctuation')
            torch.save(model.state_dict(), f'SavedModels/{path}.pth')
            return #return and end
        
    #return the model
    torch.save(model.state_dict(), f'SavedModels/{path}.pth')


#Call the train functions
#train("MainModel")
#train("Varient1")
#train("Varient2")

#Training a model on our variant 1 strucutre over 15 epoches and a tolerance of 3
train("V1Model",type=1,epochs=15,treshold=3)

"""
Post mitigation unbiased model
--------Log Output----------- Epochs 15, treshold 3, Variant 1
Epoch [1/15], Training Loss: 1.1876, Validation Loss: 1.1896
Epoch [2/15], Training Loss: 0.9700, Validation Loss: 0.9799
Epoch [3/15], Training Loss: 0.7885, Validation Loss: 0.8727
Epoch [4/15], Training Loss: 1.0400, Validation Loss: 1.0198
Epoch [5/15], Training Loss: 0.7291, Validation Loss: 1.0770
Epoch [6/15], Training Loss: 0.4977, Validation Loss: 0.9315
Epoch [7/15], Training Loss: 0.4627, Validation Loss: 1.2172
Epoch [8/15], Training Loss: 0.3522, Validation Loss: 2.4720
Epoch [9/15], Training Loss: 0.3943, Validation Loss: 1.5875
Epoch [10/15], Training Loss: 0.3870, Validation Loss: 2.0028
Epoch [11/15], Training Loss: 0.0785, Validation Loss: 1.9971
Epoch [12/15], Training Loss: 0.1436, Validation Loss: 2.9134
Epoch [13/15], Training Loss: 0.0456, Validation Loss: 3.3868
Epoch [14/15], Training Loss: 0.1650, Validation Loss: 3.2325
Epoch [15/15], Training Loss: 0.0019, Validation Loss: 4.5510
-----------------------------
"""