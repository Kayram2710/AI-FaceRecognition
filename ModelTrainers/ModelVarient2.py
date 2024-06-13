#This script will train our main model and save it in its entierty
#This varient removes convolutional layer and decreases the kernel size whitin the convultional layer
#This is to test to see if lighter computations yield worst results
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
                kernel_size = 1,    #param3: =1 to indicate that 1x1 is the size of each filter(kernels)
                #-------------------------------------------------------------------VARIENT: DECREASED SIZE
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
            )

            #-------------------------------------------------------------------VARIENT: REMOVED CONVULTIONAL LAYER

        )
        
        #Defining the flattening layer that variable "x" (1d tensor) will go through.
        self.flattenSequence = network.Sequential(

            #First call to the linearization function
            #This will flaten(connect) the learned information accross all the neurons
            network.Linear(
                #-------------------------------------------------------------------VARIENT: ADJUSTED ACCORDINGLY
                in_features = 32 * 129 * 129,   #param1: this is the size of the 1d tensor, 
                                                #its calculated by assuming 32 chanel outputs * the reduced size of the tensor from the pooling
                                                #since the images started out as 256x256 and have been halved three times after being susbtracted by 5, size of the tensor is 30x30
                out_features = 256            #param2: number of neurons in the layer 
            ),

            #Fourth call tp the activation function
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
        batch_size=40,          #param2: limit batch size to 40
        shuffle=True            #param3: set shuffle to false
    )

    #return created dataset
    return dataset

#Define function to train the model
def train(epochs = 15):

    #Create device to run model
    #This is to allow the the program to run on a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Writing parameters for early stoping procedure
    fluctuationTreshold = 2 #Patience level of 2
    countFluctuation = 0 #Count start at 0
    PreviousLoss = 0 #Create variable to store previous loss

    #Instantiate Model
    model = FaceRecognitionModel().to(device)

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
        
        #Print epoche results
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

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
            torch.save(model, 'SavedModels/Varient2.pth')
            return #return and end
        

    #return the model
    torch.save(model, 'SavedModels/Varient2.pth')

#Call the train function
train()

#This is the output from testing change by change
"""
--------Log Output----------- 
Varient: Removed Convolutional Layer Only
Epoch [1/15], Loss: 1.7341
Epoch [2/15], Loss: 1.3247
Epoch [3/15], Loss: 0.7756
Epoch [4/15], Loss: 0.5388
Epoch [5/15], Loss: 0.7531
Epoch [6/15], Loss: 0.4547
Epoch [7/15], Loss: 0.4759
Epoch [8/15], Loss: 0.4741
Epoch [9/15], Loss: 0.5552
Epoch [10/15], Loss: 0.3494
Epoch [11/15], Loss: 0.7294
Epoch [12/15], Loss: 0.4508
Epoch [13/15], Loss: 0.3626
Epoch [14/15], Loss: 0.2995
Epoch [15/15], Loss: 0.2605
-----------------------------

--------Log Output-----------
Varient: Decreased Kernel Size Only
Epoch [1/15], Loss: 1.0906
Epoch [2/15], Loss: 1.1136
Epoch [3/15], Loss: 0.9427
Epoch [4/15], Loss: 0.9449
Epoch [5/15], Loss: 0.7025
Epoch [6/15], Loss: 0.6480
Epoch [7/15], Loss: 0.8495
Epoch [8/15], Loss: 0.6618
Epoch [9/15], Loss: 0.4938
Epoch [10/15], Loss: 0.4778
Epoch [11/15], Loss: 0.3287
Epoch [12/15], Loss: 0.3291
Epoch [13/15], Loss: 0.3090
Epoch [14/15], Loss: 0.2995
Epoch [15/15], Loss: 0.1772
-----------------------------

--------Log Output----------- 
Varient: Both Removed Convolutional Layer and Decreased Kernel Size
Epoch [1/15], Loss: 6.3667
Epoch [2/15], Loss: 5.8932
Epoch [3/15], Loss: 18.3913
Epoch [4/15], Loss: 5.0900
Epoch [5/15], Loss: 3.3019
Epoch [6/15], Loss: 3.1841
Epoch [7/15], Loss: 1.3969
Epoch [8/15], Loss: 0.9858
Epoch [9/15], Loss: 1.5207
Epoch [10/15], Loss: 0.8765
Epoch [11/15], Loss: 0.4995
Epoch [12/15], Loss: 2.4214
Epoch [13/15], Loss: 1.1823
Epoch [14/15], Loss: 0.8789
Epoch [15/15], Loss: 0.6938
-----------------------------
"""