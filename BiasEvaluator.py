import numpy as np
import csv
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from ModelsScript.ModelMain import FaceRecognitionModel as Main

#Setup global variable for script
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

    return testingIndices, dataset

#Create evaluation function
def evaluate(dataset):

    model = Main().to(device)

    #Loading Model
    model.load_state_dict(
        torch.load(
            (f"SavedModels/MainModel.pth")
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
    for inputs, labels in dataset:

        #Moving tensors to the chosen device
        inputs = inputs.to(device)
        labels = labels.to(device)

        #Retrieve model prediction
        modelPrediction = model(inputs)
        _, predicted = torch.max(modelPrediction, 1)

        #Extend lists real and prediction for the computations
        prediction.extend(predicted.cpu().numpy())
        real.extend(labels.cpu().numpy())

    #Get the values using functions
    precision, recall, f1_score, _ = precision_recall_fscore_support(real, prediction, average='macro')
    accuracy = accuracy_score(real, prediction)

    #Create result dictionary
    results = {
        'Macro Precision': round(precision, 4),
        'Macro Recall': round(recall, 4),
        'Macro F1-Score': round(f1_score, 4),
        'Accuracy': round(accuracy, 4)
    }

    #Return dictionary of all scores
    return results

#Create a function to read a csv file
def readCsv(path = "DataAnnotations.csv"):

    #Open a csv reader based on path
    with open(path, mode='r') as file:

        #Instatiate reader
        reader = csv.DictReader(file)
        
        #Create empty result dict
        result = {}
        
        #Start looping through reader
        for row in reader:
            #Define key as first value in row
            key = row[reader.fieldnames[0]]
            
            #Define item as a dict filled with following rows
            item = {header: row[header] for header in reader.fieldnames[1:]}
            
            #Add to dict
            result[key] = item
            
    return result

#Creating function to run bias analysis
def runBiasEvaluation():

    #Create empty dict
    results = {}

    #Get dataset
    testingIndices, dataset = getDataset()

    #Get csv dict
    referance = readCsv()

    #Define used bias grouping
    genderGroup = ["Male","Female","Other"]
    raceGroup = ["White","Black","Asian"]
    
    #Loop for a specific gender group
    for group in genderGroup:

        #Create empty list to store selected indices
        selectedDataIndices = []

        #Loop through all indices
        for index in testingIndices:

            #Select indices if part of matches current bias group in referance
            if(referance[str(index)]["Gender"] == group):
                selectedDataIndices.append(index) #Apped to list

        #Create subset from dataset
        data = Subset(dataset, selectedDataIndices)
        #Create operationable dataset
        data = DataLoader(dataset=data,batch_size=40,shuffle=True)

        #Evaluate current group
        result = evaluate(data)
        #Add total images
        result["Number of Images"] = len(selectedDataIndices)
        #Add to results
        results[(f"Gender bias for {group}")] = result

    #Add averages to dict    
    results = getAverage(results,"Gender")

    #Print results
    for key, item in results.items():
        print(f"{key}:\n{item}\n")
    
    #Create empty dict
    results = {}

    #Loop for a specific race group
    for group in raceGroup:

        #Create empty list to store selected indices
        selectedDataIndices = []

        #Loop through all indices
        for index in testingIndices:

            #Select indices if part of matches current bias group in referance
            if(referance[str(index)]["Race"] == group):
                selectedDataIndices.append(index) #Apped to list

        #Create subset from dataset
        data = Subset(dataset, selectedDataIndices)
        #Create operationable dataset
        data = DataLoader(dataset=data,batch_size=40,shuffle=True)

        #Evaluate current group
        result = evaluate(data)
        #Add total images
        result["Number of Images"] = len(selectedDataIndices)
        #Add to results
        results[(f"Race bias for {group}")] = result

    #Add averages to dict    
    results = getAverage(results,"Race")

    #Print results
    for key, item in results.items():
        print(f"{key}:\n{item}\n")

    #Return results
    return results

#Create get average function
def getAverage(results,type):

    #Setup total list
    totals = [0,0,0,0,0]

    #Setup divisor
    divisor = 3

    #Fill list up
    for _, item in results.items():
        for i ,(_, score) in enumerate(item.items()):
            totals[i] += score

    #Updated dict entry for averages
    results[f'Averages/Totals for {type} bias'] = {
        'Macro Precision': round(totals[0]/divisor, 4),
        'Macro Recall': round(totals[1]/divisor, 4),
        'Macro F1-Score': round(totals[2]/divisor, 4),
        'Accuracy': round(totals[3]/divisor, 4),
        'Total Images': totals[4]
    }

    #Return results
    return results

runBiasEvaluation()

"""
--------Log Output-----------
Gender bias for Male:
{'Macro Precision': 0.5993, 'Macro Recall': 0.5946, 'Macro F1-Score': 0.5969, 'Accuracy': 0.5698, 'Number of Images': 179}

Gender bias for Female:
{'Macro Precision': 0.5464, 'Macro Recall': 0.5201, 'Macro F1-Score': 0.5287, 'Accuracy': 0.5778, 'Number of Images': 135}

Gender bias for Other:
{'Macro Precision': 0.4345, 'Macro Recall': 0.5038, 'Macro F1-Score': 0.4533, 'Accuracy': 0.3784, 'Number of Images': 37}

Averages/Totals for Gender bias:
{'Macro Precision': 0.5267, 'Macro Recall': 0.5395, 'Macro F1-Score': 0.5263, 'Accuracy': 0.5087, 'Total Images': 351}

Race bias for White:
{'Macro Precision': 0.5841, 'Macro Recall': 0.5639, 'Macro F1-Score': 0.5718, 'Accuracy': 0.5556, 'Number of Images': 252}

Race bias for Black:
{'Macro Precision': 0.5816, 'Macro Recall': 0.574, 'Macro F1-Score': 0.5643, 'Accuracy': 0.5806, 'Number of Images': 62}

Race bias for Asian:
{'Macro Precision': 0.6027, 'Macro Recall': 0.5833, 'Macro F1-Score': 0.5583, 'Accuracy': 0.5, 'Number of Images': 30}

Race bias for Native:
{'Macro Precision': 0.375, 'Macro Recall': 0.3333, 'Macro F1-Score': 0.35, 'Accuracy': 0.4286, 'Number of Images': 7}

Averages/Totals for Race bias:
{'Macro Precision': 0.5358, 'Macro Recall': 0.5136, 'Macro F1-Score': 0.5111, 'Accuracy': 0.5162, 'Total Images': 351}
-----------------------------
"""
