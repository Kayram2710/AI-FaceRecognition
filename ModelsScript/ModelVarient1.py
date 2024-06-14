#This script will train our main model and save it in its entierty
#This varient adds a third convolutional layer and increases the kernel size whitin the convultional layers
#This is to test to see if heavier computations yield better results
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
                kernel_size = 5,    #param3: =5 to indicate that 5x5 is the size of each filter(kernels)
                #-------------------------------------------------------------------VARIENT: INCREASED SIZE
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
                kernel_size = 5,    #param3: =5 to indicate that 5x5 is the size of each filter(kernels)
                #-------------------------------------------------------------------VARIENT: INCREASED SIZE
                padding = 1         #param4: =1 to add a padding of width 1 to the border of the image 
            ),

            #Second call to the activation function
            network.ReLU(),

            #Call to the second pooling layer
            network.MaxPool2d(
                kernel_size=2,  #param1: =2 to indicate that 2x2 is the size of the reduced filters(kernels)
                stride=2        #param2: =2 to indicate the process to move 2 windows at a time while grouping
            ),

            #-------------------------------------------------------------------VARIENT: ADDED THIRD CONVULTIONAL LAYER
            #Call to the third convolutional layer
            network.Conv2d(
                in_channels= 64,    #param1: =64 indicates the amount of input channels, changed by the previous convolutional layer
                out_channels= 128,  #param2: =128 is the number of filters(kernels) the input("x") will go through
                kernel_size = 5,    #param3: =5 to indicate that 5x5 is the size of each filter(kernels)
                #-------------------------------------------------------------------VARIENT: INCREASED SIZE
                padding = 1         #param4: =1 to add a padding of width 1 to the border of the image 
            ),

            #Third call to the activation function
            network.ReLU(),

            #Call to the third pooling layer
            network.MaxPool2d(
                kernel_size=2,  #param1: =2 to indicate that 2x2 is the size of the reduced filters(kernels)
                stride=2        #param2: =2 to indicate the process to move 2 windows at a time while grouping
            )

        )
        
        #Defining the flattening layer that variable "x" (1d tensor) will go through.
        self.flattenSequence = network.Sequential(

            #First call to the linearization function
            #This will flaten(connect) the learned information accross all the neurons
            #-------------------------------------------------------------------VARIENT: ADJUSTED ACCORDINGLY
            network.Linear(
                in_features = 128 * 30 * 30,   #param1: this is the size of the 1d tensor, 
                                                #its calculated by assuming 128 chanel outputs * the reduced size of the tensor from the pooling
                                                #since the images started out as 256x256 and have been halved three times after being susbtracted by 5, size of the tensor is 30x30
                out_features = 256             #param2: number of neurons in the layer 
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
Varient: Additional Convolutional Layer Only
Epoch [1/15], Loss: 1.0907
Epoch [2/15], Loss: 1.0015
Epoch [3/15], Loss: 0.9681
Epoch [4/15], Loss: 0.7922
Epoch [5/15], Loss: 0.6108
Epoch [6/15], Loss: 0.4133
Epoch [7/15], Loss: 0.3000
Epoch [8/15], Loss: 0.3588
Epoch [9/15], Loss: 0.2410
Epoch [10/15], Loss: 0.0421
Epoch [11/15], Loss: 0.0208
Epoch [12/15], Loss: 0.0135
Epoch [13/15], Loss: 0.0629
Epoch [14/15], Loss: 0.0067
Epoch [15/15], Loss: 0.0165
-----------------------------

--------Log Output-----------
Varient: Increased Kernel Size Only
Epoch [1/15], Loss: 1.0391
Epoch [2/15], Loss: 0.7585
Epoch [3/15], Loss: 0.7005
Epoch [4/15], Loss: 0.7882
Epoch [5/15], Loss: 0.4910
Epoch [6/15], Loss: 0.7140
Epoch [7/15], Loss: 0.5465
Epoch [8/15], Loss: 0.2754
Epoch [9/15], Loss: 0.2429
Epoch [10/15], Loss: 0.1275
Epoch [11/15], Loss: 0.0630
Epoch [12/15], Loss: 0.0510
Epoch [13/15], Loss: 0.0635
Epoch [14/15], Loss: 0.0075
Epoch [15/15], Loss: 0.0619
-----------------------------

--------Log Output----------- 
Varient: Both Additional Convolutional Layer and Increased Kernel Size
Epoch [1/15], Loss: 0.9106
Epoch [2/15], Loss: 0.8368
Epoch [3/15], Loss: 0.8305
Epoch [4/15], Loss: 0.8134
Epoch [5/15], Loss: 0.7043
Epoch [6/15], Loss: 0.7611
Epoch [7/15], Loss: 0.4644
Epoch [8/15], Loss: 0.3357
Epoch [9/15], Loss: 0.2312
Epoch [10/15], Loss: 0.1233
Epoch [11/15], Loss: 0.0579
Epoch [12/15], Loss: 0.0345
Epoch [13/15], Loss: 0.0314
Epoch [14/15], Loss: 0.0748
Epoch [15/15], Loss: 0.0027
-----------------------------
"""