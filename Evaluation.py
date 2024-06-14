import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

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



def evaluate(model):

    dataloader = getDataset()

    #Create device to run model
    #This is to allow the the program to run on a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Calculate metrics
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)

    return {
        'Macro Precision': precision,
        'Macro Recall': recall,
        'Macro F1-Score': f1_score,
        'Micro Precision': micro_precision,
        'Micro Recall': micro_recall,
        'Micro F1-Score': micro_f1,
        'Accuracy': accuracy
    }

#Loading the existing models
mainModel = torch.load("SavedModels/MainModel.pth")
varient1 = torch.load("SavedModels/Varient1.pth")
varient2 = torch.load("SavedModels/Varient2.pth")

print(evaluate(mainModel))