#This script will train our main model and save it in its entierty
import torch.nn as network

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
                                                #since the images started out as 256x256 and have been halved twice after being substracted by 3, the size of the tensor is 64x64
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

#This is the output
"""
--------Log Output-----------
Epoch [1/15], Training Loss: 1.2089, Validation Loss: 1.1799
Epoch [2/15], Training Loss: 0.8870, Validation Loss: 0.9139
Epoch [3/15], Training Loss: 0.8267, Validation Loss: 0.9611
Epoch [4/15], Training Loss: 0.6360, Validation Loss: 0.8846
Epoch [5/15], Training Loss: 0.6449, Validation Loss: 0.8173
Epoch [6/15], Training Loss: 0.4758, Validation Loss: 0.9664
Epoch [7/15], Training Loss: 0.3062, Validation Loss: 0.8261
Epoch [8/15], Training Loss: 0.1860, Validation Loss: 1.7813
Epoch [9/15], Training Loss: 0.1190, Validation Loss: 0.9673
Epoch [10/15], Training Loss: 0.0671, Validation Loss: 1.4820
Epoch [11/15], Training Loss: 0.0395, Validation Loss: 1.1336
Epoch [12/15], Training Loss: 0.0279, Validation Loss: 1.9258
Epoch [13/15], Training Loss: 0.0148, Validation Loss: 1.9998
Exited program due to too many fluctuation
-----------------------------
"""