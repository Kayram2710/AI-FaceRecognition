import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from ModelsScript.ModelMain import FaceRecognitionModel as Main
from ModelsScript.ModelVarient1 import FaceRecognitionModel as V1
from ModelsScript.ModelVarient2 import FaceRecognitionModel as V2

#Create device to run model
#This is to allow the the program to run on a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    #Splitting the data to retrieve testing group
    _, testingIndices = train_test_split(
        np.arange(len(dataset.targets)),    #param1: list of all indices in the dataset
        test_size=0.15,                     #param2: portion of split data (15%)
        random_state=50,                    #param3: seed for shuffling
        shuffle=True,                       #param4: enable shuffling
        stratify=dataset.targets            #param5: strasfying in order to have proportionate types of labels
    )

    #Creating the datasets based off the indices generated 
    testingData = Subset(dataset, testingIndices) #Creating testing set using testing indices

    #Create operationable dataset
    testingData = DataLoader(  #Call data loader to load training data
        dataset=testingData,   #param1: dataset points to a newly created dataset "testing data"          
        batch_size=40,         #param2: limit batch size to 40
        shuffle=True           #param3: set shuffle to True
    )

    #return created dataset
    return testingData

#Create a function too chose a structure
def chooseStructure(path):
    
    state_dict = torch.load(path)
    testingVar = str((state_dict["initialSequence.0.weight"]).shape)

    #Instantiate Model Depending on path chosen
    if(testingVar == "torch.Size([32, 1, 5, 5])"):
        model = V1().to(device)
    elif(testingVar == "torch.Size([32, 1, 1, 1])"):
        model = V2().to(device)
    elif(testingVar == "torch.Size([32, 1, 3, 3])"):
       model = Main().to(device)

    return model


#Create evaluation function
def evaluate(name):

    #Define ful path
    path = f"SavedModels/{name}.pth"

    #Get Testing Data
    testingData = getDataset()

    #Check if path exists
    if not os.path.exists(path):
        return False #Does not allow for a toss up path
    
    model = chooseStructure(path)

    #Loading Model
    model.load_state_dict(
        torch.load(
            (path)
            #, map_location=torch.device('cpu') #Uncomment if running on cpu
            ))
    
    #Send it to selected device
    model.to(device)

    #Setting evaluation mode
    model.eval()
    
    #Define empty lists
    real = []
    prediction = []

    #For loop through each image in the validation dataset and return the images and their label
    for inputs, labels in testingData:

        #Moving tensors to the chosen device
        inputs = inputs.to(device)
        labels = labels.to(device)

        #Retrieve model prediction
        modelPrediction = model(inputs)
        _, predicted = torch.max(modelPrediction, 1)

        #Extend lists real and prediction for the computations
        prediction.extend(predicted.cpu().numpy())
        real.extend(labels.cpu().numpy())

    #Generate confusion matrix based on axis
    cmMatrix = confusion_matrix(real, prediction)

    #Generate labels
    labels = ["Angry", "Focused","Happy", "Neutral"]

    #Generate Display
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=cmMatrix, display_labels=labels)

    #Plot display
    display.plot()
    plt.show()

    #Get the values using functions
    precision, recall, f1_score, _ = precision_recall_fscore_support(real, prediction, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(real, prediction, average='micro')
    accuracy = accuracy_score(real, prediction)

    #Create result dictionary
    results = {
        'Macro Precision': precision,
        'Macro Recall': recall,
        'Macro F1-Score': f1_score,
        'Micro Precision': micro_precision,
        'Micro Recall': micro_recall,
        'Micro F1-Score': micro_f1,
        'Accuracy': accuracy
    }

    #Log feedback
    print(f"Result for {name}:\n{results}")

    #Return dictionary of all scores
    return results

#evaluate("MainModel")
#evaluate("Varient1")
#evaluate("Varient2")
evaluate("CurrentBest")

"""
--------Log Output Pre Mitigation-----------
Result for MainModel:
{'Macro Precision': 0.5812893612235718, 'Macro Recall': 0.565018315018315, 'Macro F1-Score': 0.5717070799413753, 'Micro Precision': 0.5527065527065527, 'Micro Recall': 0.5527065527065527, 'Micro F1-Score': 0.5527065527065527, 'Accuracy': 0.5527065527065527}
Result for Varient1:
{'Macro Precision': 0.6127269848713761, 'Macro Recall': 0.6108058608058607, 'Macro F1-Score': 0.6096439129276968, 'Micro Precision': 0.5982905982905983, 'Micro Recall': 0.5982905982905983, 'Micro F1-Score': 0.5982905982905983, 'Accuracy': 0.5982905982905983}
Result for Varient2:
{'Macro Precision': 0.5595254874437258, 'Macro Recall': 0.5329670329670331, 'Macro F1-Score': 0.51985658800616, 'Micro Precision': 0.5242165242165242, 'Micro Recall': 0.5242165242165242, 'Micro F1-Score': 0.5242165242165242, 'Accuracy': 0.5242165242165242}
-----------------------------
--------Log Output Post Mitigation for "CurrentBest"-----------
Result for CurrentBest:
{'Macro Precision': 0.566968892820286, 'Macro Recall': 0.5454473306446991, 'Macro F1-Score': 0.5513555682036281, 'Micro Precision': 0.5355329949238579, 'Micro Recall': 0.5355329949238579, 'Micro F1-Score': 0.5355329949238579, 'Accuracy': 0.5355329949238579}
-----------------------------
"""