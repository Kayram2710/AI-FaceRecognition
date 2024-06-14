#This script will train our main model and save it in its entierty
#This varient removes convolutional layer and decreases the kernel size whitin the convultional layer
#This is to test to see if lighter computations yield worst results
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
Epoch [1/15], Training Loss: 4.0488, Validation Loss: 3.1089
Epoch [2/15], Training Loss: 1.5199, Validation Loss: 1.3053
Epoch [3/15], Training Loss: 1.6142, Validation Loss: 1.2135
Epoch [4/15], Training Loss: 1.1551, Validation Loss: 1.5750
Epoch [5/15], Training Loss: 1.1071, Validation Loss: 1.3893
Epoch [6/15], Training Loss: 0.6416, Validation Loss: 0.7675
Epoch [7/15], Training Loss: 1.2026, Validation Loss: 1.7823
Epoch [8/15], Training Loss: 1.0687, Validation Loss: 1.2099
Epoch [9/15], Training Loss: 1.2628, Validation Loss: 1.7213
Epoch [10/15], Training Loss: 1.0916, Validation Loss: 1.0847
Epoch [11/15], Training Loss: 0.7016, Validation Loss: 1.2283
Epoch [12/15], Training Loss: 0.9602, Validation Loss: 1.0593
Epoch [13/15], Training Loss: 0.6584, Validation Loss: 1.2288
Epoch [14/15], Training Loss: 0.9142, Validation Loss: 1.1670
Epoch [15/15], Training Loss: 0.8069, Validation Loss: 1.0080
-----------------------------
"""