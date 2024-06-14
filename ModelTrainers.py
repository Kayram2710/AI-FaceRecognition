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
def train(path, epochs = 15):

    #Create device to run model
    #This is to allow the the program to run on a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Writing parameters for early stoping procedure
    fluctuationTreshold = 2 #Patience level of 2
    countFluctuation = 0 #Count start at 0
    PreviousLoss = 0 #Create variable to store previous loss

    #Instantiate Model Depending on path fed
    if(path == "Varient1"):
        model = V1().to(device)
    elif(path == "Varient2"):
        model = V2().to(device)
    else:
        model = Main().to(device)

    #Create a call for the "Cross Entrop Loss" function
    #This function will calculate the differance across 
    criterion = network.CrossEntropyLoss()

    #Setting an Adam optimizer
    #This will allow the model to adjust itself based on the returned loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Retrieve Dataset
    dataset, valDataset = getDataset()

    #Setting model to training mode
    model.train()

    #Start looping through epochs
    for epoch in range(epochs):

        #For loop through each image in the dataset and return the images and theyre label
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

        # Validation phase ----------------------------------
        model.eval()
        with torch.no_grad():
            for images, labels in valDataset:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
        
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

        #Early stop procedure
        #If epoch is at 0, do nothing
        if(epoch == 0):
            ""
        #Else check if new loss is greater then previous
        elif(loss.item() >= PreviousLoss):
            countFluctuation = 1 + countFluctuation #If it is count the fluctuation
        else:
            countFluctuation = 0 #If it isnt reset it
        
        #Note down previous loss
        PreviousLoss = loss.item()

        #Then check if count equal or surpasses treshold
        if(countFluctuation >= fluctuationTreshold):
            print(f'Exited program due to too many fluctuation')
            torch.save(model, f'SavedModels/{path}.pth')
            return #return and end
        

    #return the model
    torch.save(model, f'SavedModels/{path}.pth')


#Call the train function
train("Test")
