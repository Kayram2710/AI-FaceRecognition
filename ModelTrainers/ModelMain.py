#This script will train our main model and save it in its entierty
import torch
import torch.nn as network
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#-------------------------------------------------------------------------------------------------------
#Beggining of the architrecture of our CNN with torough explanation in order to aid at writing the report 
#Creating class for the Cnetwork
class FaceRecognitionModel(network.Module):
    
    #Constructor
    def __init__(self):
        
        #Calling parent consructor
        super(FaceRecognitionModel, self).__init__()

        #Defining the initial sequence of layers that variable "x" (2d tensor) will go through.
        #A "network.Sequential" variable works like a function(def) variable, 
        #It's use, is that does not require redefining the variable "x" through each layer call (turns "x = func(x)" call into "func").
        self.initialSequence = network.Sequential(

            #Call to first convolutional layer
            #This layer will put the data through kernels in order to analyze it
            network.Conv2d(
                in_channels= 1,     #param1: =1 indicates input("x") has one input channel since the data is in grayscale
                out_channels= 32,   #param2: =32 is the number of filters(kernels) the input("x") will go through
                kernel_size = 3,    #param3: =3 to indicate that 3x3 is the size of each filter(kernels)
                padding = 1         #param4: =1 to add a padding of width 1 to the border of the image 
            ),

            #First call to the activation function
            # "Introduces non linearity ot the model in order for it to learn more complex patern" - Source: Gpt4
            network.ReLU(),

            #Call to the first pooling layer
            #This layer will reduce the extracted data into groups where it will pick the max value from each group to save
            network.MaxPool2d(
                kernel_size=2,  #param1: =2 to indicate that 2x2 is the size of the reduced filters(kernels)
                stride=2        #param2: =2 to indicate the process to move 2 windows at a time while grouping
            ),

            #Call to the second convolutional layer
            network.Conv2d(
                in_channels= 32,    #param1: =32 indicates the amount of input channels, changed by the previous convolutional layer
                out_channels= 64,   #param2: =64 is the number of filters(kernels) the input("x") will go through
                kernel_size = 3,    #param3: =3 to indicate that 3x3 is the size of each filter(kernels)
                padding = 1         #param4: =1 to add a padding of width 1 to the border of the image 
            ),

            #Second call to the activation function
            network.ReLU(),

            #Call to the second pooling layer
            network.MaxPool2d(
                kernel_size=2,  #param1: =2 to indicate that 2x2 is the size of the reduced filters(kernels)
                stride=2        #param2: =2 to indicate the process to move 2 windows at a time while grouping
            )
        )
        
        #Defining the flattening layer that variable "x" (1d tensor) will go through.
        self.flattenSequence = network.Sequential(

            #First call to the linearization function
            #This will flaten(connect) the learned information accross all the neurons
            network.Linear(
                in_features = 64 * 64 * 64,   #param1: this is the size of the 1d tensor, 
                                                #its calculated by assuming 64 chanel outputs * the reduced size of the tensor from the pooling
                                                #since the images started out as 256x256 and have been halved twice the size of the tensor is 64x64
                out_features = 256             #param2: number of neurons in the layer 
            ),

            #Third call tp the activation function
            network.ReLU(),

            #Second call to the linearization function
            #This call will flaten the remaining neurons into 4 classes
            network.Linear(
                in_features = 256,    #param1: this value is the amount of neurons in the model
                out_features = 4      #param2: this is the amount of classification the model is training for (4 emotios to recognize)
            )
        )
        
    #Once called this function will essentially act as the driver for it's instantiated object
    def forward(self, x):
        #Running x through the sequencial calls defined by initialSequence
        x = self.initialSequence(x)

        #Flatening the tensor into a 1 dimensional array
        x = x.view(x.size(0), -1)

        #Running x through the sequencial calls defined by second sequence
        x = self.flattenSequence(x)

        return x #return tensor(x)
    
#End of CNN Architecture ----------------------------------------------------------------------------------------------

#Define function for returning dataset
def getDataset(path = "Cleaned Dataset"):

    #Defining transform operations
    transform = transforms.Compose([
        transforms.Grayscale(1), #Grayscale
        transforms.ToTensor(), #Turn data into readable tensors          
        transforms.Normalize((0.5,), (0.5,)) #Normalize data
    ])

    #Create variable to existing dataset based on an image folder
    trainingData = datasets.ImageFolder(   
        root=path,              #param1: path to dataset
        transform=transform     #param2: point to transform operation
    )

    #Create operationable dataset
    dataset = DataLoader(       #Call data loader to load training data
        dataset=trainingData,   #param1: dataset points to a newly created dataset "training data"          
        batch_size=400,         #param2: limit batch size to 400
        shuffle=False           #param3: set shuffle to false
    )

    #return created dataset
    return dataset

#Define function to train the model
def train(epochs = 10):

    #Writing parameters for early stoping procedure
    fluctuationTreshold = 2 #Patience level of 2
    countFluctuation = 0 #Count start at 0
    PreviousLoss = 0 #Create variable to store previous loss

    #Instantiate Model
    model = FaceRecognitionModel()

    #Create a call for the "Cross Entrop Loss" function
    #This function will calculate the differance across 
    criterion = network.CrossEntropyLoss()

    #Setting an Adam optimizer
    #This will allow the model to adjust itself based on the returned loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Retrieve Dataset
    dataset = getDataset()

    #Setting model to training mode
    model.train()

    #Start looping through epochs
    for epoch in range(epochs):

        #For loop through each image in the dataset and return the images and theyre label
        for images, labels in dataset:

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
        
        #Print epoche results
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        #Early stop procedure
        #If epoch is at 0, do nothing
        if(epoch == 0):
            PreviousLoss = loss.item()
        #Else check if new loss is greater then previous
        elif(loss.item >= PreviousLoss):
            countFluctuation = 1 + countFluctuation #If it is count the fluctuation
        else:
            countFluctuation = 0 #If it isnt reset it
        
        #Then check if count equal or surpasses treshold
        if(countFluctuation >= fluctuationTreshold):
            print(f'Exited program as model was faulty')
            return #return and end


    #return the model
    torch.save(model, 'SavedModels/MainModel.pth')

#Call the train function
train()

#This is the output
"""
--------Log Output-----------
Epoch [1/10], Loss: 10.5292
Epoch [2/10], Loss: 1.4596
Epoch [3/10], Loss: 1.4481
Epoch [4/10], Loss: 1.4089
Epoch [5/10], Loss: 1.4085
Epoch [6/10], Loss: 1.4077
Epoch [7/10], Loss: 1.4071
Epoch [8/10], Loss: 1.4069
Epoch [9/10], Loss: 1.4060
Epoch [10/10], Loss: 1.4056
-----------------------------
"""