import torch
import torch.nn as network
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold, train_test_split
from ModelsScript.ModelMain import FaceRecognitionModel as Main

#Setup global variable for script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Define function for returning dataset
def getDatasetIndex(path = "Cleaned Dataset"):

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

    #return created dataset
    return dataset


#Create evaluation function
def evaluate(model, dataset):
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
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(real, prediction, average='micro')
    accuracy = accuracy_score(real, prediction)

    #Create result dictionary
    results = {
        'Macro Precision': round(precision, 4),
        'Macro Recall': round(recall, 4),
        'Macro F1-Score': round(f1_score, 4),
        'Micro Precision': round(micro_precision, 4),
        'Micro Recall': round(micro_recall, 4),
        'Micro F1-Score': round(micro_f1, 4),
        'Accuracy': round(accuracy, 4)
    }

    #Log feedback
    print(f"Result:\n{results}")

    #Return dictionary of all scores
    return results

#Define function to train the model
def train(model, trainingData, validationData):
    #Deciding on 15 epochs
    epochs = 15

    #Writing parameters for early stoping procedure
    fluctuationTreshold = 2 #Patience level of 2
    countFluctuation = 0 #Count start at 0
    PreviousLoss = 0 #Create variable to store previous loss

    #Create a call for the "Cross Entrop Loss" function
    #This function will calculate the differance across 
    criterion = network.CrossEntropyLoss()

    #Setting an Adam optimizer
    #This will allow the model to adjust itself based on the returned loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Start looping through epochs
    for epoch in range(epochs):

        #Setting model to training mode
        model.train()

        #For loop through each image in the dataset and return the images and their label
        for images, labels in trainingData:

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
        for images, labels in validationData:

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
            return model #return and end
    
    return model

#Creating function to run Kfold
def runKfold(folds=10):

    #Setting up Kfold
    kf = KFold(n_splits=folds, shuffle=True, random_state=50)
    results = {}

    #Retrive dasetIndices
    datasetIndices = getDatasetIndex()

    #Start enumerated loop for kfold
    for fold, (trainIndices, testIndices) in enumerate(kf.split(datasetIndices)):

        #Instatiate Model
        model = Main().to(device)

        #Splitting the data to retrieve testing group
        trainIndices, validationIndices = train_test_split(
            trainIndices,                                                   #param1: list of all indices in the dataset
            test_size=0.15,                                                 #param2: portion of split data (15%)
            random_state=50,                                                #param3: seed for shuffling
            shuffle=True,                                                   #param4: enable shuffling
            stratify=[datasetIndices.targets[i] for i in trainIndices]      #param5: strasfying in order to have proportionate types of labels
        )

        #Creating Subsets
        trainingData = Subset(datasetIndices, trainIndices) #Creating training set using training indices
        validationData = Subset(datasetIndices, validationIndices) #Creating validation set using validation indices
        testingData = Subset(datasetIndices, testIndices) #Creating testing data using testing indices
        
        #Create operationable datasets
        trainingData = DataLoader(dataset=trainingData,batch_size=40,shuffle=True)
        validationData = DataLoader(dataset=validationData,batch_size=40,shuffle=True)
        testingData = DataLoader(dataset=testingData,batch_size=40,shuffle=True)

        #Train model
        model = train(model, trainingData, validationData)

        #Evaluate model
        result = evaluate(model, testingData)

        #Log to dict "results"
        results[(f"Fold {fold+1}")] = result

    #Add averages to dict    
    results = getAverage(results)

    #Print results
    for key, item in results.items():
        print(f"{key}:\n{item}\n")

    #Return results
    return results

#Create get average function
def getAverage(results):

    #Setup total list
    totals = [0,0,0,0,0,0,0]

    #Fill list up
    for _, item in results.items():
        for i ,(_, score) in enumerate(item.items()):
            totals[i] += score

    #Updated dict entry for averages
    results['Averages'] = {
        'Macro Precision': round(totals[0]/10, 4),
        'Macro Recall': round(totals[1]/10, 4),
        'Macro F1-Score': round(totals[2]/10, 4),
        'Micro Precision': round(totals[3]/10, 4),
        'Micro Recall': round(totals[4]/10, 4),
        'Micro F1-Score': round(totals[5]/10, 4),
        'Accuracy': round(totals[6]/10, 4)
    }

    #Return results
    return results

runKfold()

"""
--------Log Output-----------
Epoch [1/15], Training Loss: 0.9827, Validation Loss: 1.0847
Epoch [2/15], Training Loss: 0.8479, Validation Loss: 0.7887
Epoch [3/15], Training Loss: 0.6408, Validation Loss: 0.9857
Epoch [4/15], Training Loss: 0.4334, Validation Loss: 0.9965
End of training due to too many fluctuation
Result:
{'Macro Precision': 0.6282, 'Macro Recall': 0.5974, 'Macro F1-Score': 0.6077, 'Micro Precision': 0.5855, 'Micro Recall': 0.5855, 'Micro F1-Score': 0.5855, 'Accuracy': 0.5855}
Epoch [1/15], Training Loss: 0.9114, Validation Loss: 0.8918
Epoch [2/15], Training Loss: 0.6797, Validation Loss: 0.8552
Epoch [3/15], Training Loss: 0.8571, Validation Loss: 0.7139
Epoch [4/15], Training Loss: 0.7937, Validation Loss: 0.8836
Epoch [5/15], Training Loss: 0.7740, Validation Loss: 1.0676
End of training due to too many fluctuation
Result:
{'Macro Precision': 0.6066, 'Macro Recall': 0.5863, 'Macro F1-Score': 0.5856, 'Micro Precision': 0.5684, 'Micro Recall': 0.5684, 'Micro F1-Score': 0.5684, 'Accuracy': 0.5684}
Epoch [1/15], Training Loss: 1.0557, Validation Loss: 0.9890
Epoch [2/15], Training Loss: 1.0500, Validation Loss: 0.9707
Epoch [3/15], Training Loss: 0.7499, Validation Loss: 0.8955
Epoch [4/15], Training Loss: 0.5187, Validation Loss: 1.1713
Epoch [5/15], Training Loss: 0.5462, Validation Loss: 1.0581
Epoch [6/15], Training Loss: 0.2438, Validation Loss: 1.4640
Epoch [7/15], Training Loss: 0.2638, Validation Loss: 1.4586
Epoch [8/15], Training Loss: 0.0690, Validation Loss: 2.5121
Epoch [9/15], Training Loss: 0.0913, Validation Loss: 1.7306
Epoch [10/15], Training Loss: 0.0233, Validation Loss: 2.0900
Epoch [11/15], Training Loss: 0.0020, Validation Loss: 3.0043
End of training due to too many fluctuation
Result:
{'Macro Precision': 0.6008, 'Macro Recall': 0.6001, 'Macro F1-Score': 0.6001, 'Micro Precision': 0.5983, 'Micro Recall': 0.5983, 'Micro F1-Score': 0.5983, 'Accuracy': 0.5983}
Epoch [1/15], Training Loss: 1.1543, Validation Loss: 0.9806
Epoch [2/15], Training Loss: 0.8933, Validation Loss: 0.7466
Epoch [3/15], Training Loss: 0.9040, Validation Loss: 1.1343
Epoch [4/15], Training Loss: 0.5190, Validation Loss: 0.8741
Epoch [5/15], Training Loss: 0.3355, Validation Loss: 0.8544
Epoch [6/15], Training Loss: 0.2248, Validation Loss: 1.6680
Epoch [7/15], Training Loss: 0.1157, Validation Loss: 1.0212
Epoch [8/15], Training Loss: 0.0232, Validation Loss: 2.3942
Epoch [9/15], Training Loss: 0.0179, Validation Loss: 1.4311
Epoch [10/15], Training Loss: 0.0043, Validation Loss: 2.6930
Epoch [11/15], Training Loss: 0.0019, Validation Loss: 2.3731
Epoch [12/15], Training Loss: 0.0004, Validation Loss: 2.7023
Epoch [13/15], Training Loss: 0.0012, Validation Loss: 3.6639
End of training due to too many fluctuation
Result:
{'Macro Precision': 0.5721, 'Macro Recall': 0.569, 'Macro F1-Score': 0.5651, 'Micro Precision': 0.547, 'Micro Recall': 0.547, 'Micro F1-Score': 0.547, 'Accuracy': 0.547}
Epoch [1/15], Training Loss: 1.1076, Validation Loss: 1.1266
Epoch [2/15], Training Loss: 0.8827, Validation Loss: 1.1250
Epoch [3/15], Training Loss: 0.9319, Validation Loss: 0.8605
Epoch [4/15], Training Loss: 0.7860, Validation Loss: 1.0983
Epoch [5/15], Training Loss: 0.9223, Validation Loss: 0.9029
Epoch [6/15], Training Loss: 0.2742, Validation Loss: 0.9516
Epoch [7/15], Training Loss: 0.2705, Validation Loss: 1.7317
End of training due to too many fluctuation
Result:
{'Macro Precision': 0.6051, 'Macro Recall': 0.6002, 'Macro F1-Score': 0.5955, 'Micro Precision': 0.5769, 'Micro Recall': 0.5769, 'Micro F1-Score': 0.5769, 'Accuracy': 0.5769}
Epoch [1/15], Training Loss: 1.2828, Validation Loss: 1.2613
Epoch [2/15], Training Loss: 0.9256, Validation Loss: 1.1958
Epoch [3/15], Training Loss: 0.7307, Validation Loss: 1.0777
Epoch [4/15], Training Loss: 0.9770, Validation Loss: 1.0342
Epoch [5/15], Training Loss: 0.5771, Validation Loss: 1.1921
Epoch [6/15], Training Loss: 0.4125, Validation Loss: 1.1606
Epoch [7/15], Training Loss: 0.1562, Validation Loss: 1.9058
Epoch [8/15], Training Loss: 0.1648, Validation Loss: 1.3696
Epoch [9/15], Training Loss: 0.0219, Validation Loss: 1.8210
Epoch [10/15], Training Loss: 0.0310, Validation Loss: 1.4928
Epoch [11/15], Training Loss: 0.0204, Validation Loss: 2.7714
Epoch [12/15], Training Loss: 0.0420, Validation Loss: 2.5757
Epoch [13/15], Training Loss: 0.0199, Validation Loss: 2.5818
Epoch [14/15], Training Loss: 0.0037, Validation Loss: 2.0291
Epoch [15/15], Training Loss: 0.0018, Validation Loss: 2.3266
Result:
{'Macro Precision': 0.6219, 'Macro Recall': 0.6203, 'Macro F1-Score': 0.6175, 'Micro Precision': 0.6309, 'Micro Recall': 0.6309, 'Micro F1-Score': 0.6309, 'Accuracy': 0.6309}
Epoch [1/15], Training Loss: 0.8134, Validation Loss: 0.9932
Epoch [2/15], Training Loss: 0.8065, Validation Loss: 0.7901
Epoch [3/15], Training Loss: 0.9721, Validation Loss: 0.8687
Epoch [4/15], Training Loss: 0.6594, Validation Loss: 0.9481
End of training due to too many fluctuation
Result:
{'Macro Precision': 0.5601, 'Macro Recall': 0.5565, 'Macro F1-Score': 0.5398, 'Micro Precision': 0.5408, 'Micro Recall': 0.5408, 'Micro F1-Score': 0.5408, 'Accuracy': 0.5408}
Epoch [1/15], Training Loss: 1.1317, Validation Loss: 1.2705
Epoch [2/15], Training Loss: 1.0322, Validation Loss: 1.1512
Epoch [3/15], Training Loss: 0.6652, Validation Loss: 0.9935
Epoch [4/15], Training Loss: 1.2330, Validation Loss: 1.4379
Epoch [5/15], Training Loss: 1.0074, Validation Loss: 0.8142
Epoch [6/15], Training Loss: 0.7260, Validation Loss: 0.9211
Epoch [7/15], Training Loss: 0.8226, Validation Loss: 1.0723
End of training due to too many fluctuation
Result:
{'Macro Precision': 0.5277, 'Macro Recall': 0.5086, 'Macro F1-Score': 0.4835, 'Micro Precision': 0.4979, 'Micro Recall': 0.4979, 'Micro F1-Score': 0.4979, 'Accuracy': 0.4979}
Epoch [1/15], Training Loss: 1.1695, Validation Loss: 1.0459
Epoch [2/15], Training Loss: 0.7903, Validation Loss: 0.8249
Epoch [3/15], Training Loss: 0.7123, Validation Loss: 0.8601
Epoch [4/15], Training Loss: 0.7532, Validation Loss: 1.0715
End of training due to too many fluctuation
Result:
{'Macro Precision': 0.6585, 'Macro Recall': 0.5941, 'Macro F1-Score': 0.579, 'Micro Precision': 0.5579, 'Micro Recall': 0.5579, 'Micro F1-Score': 0.5579, 'Accuracy': 0.5579}
Epoch [1/15], Training Loss: 1.1899, Validation Loss: 1.1260
Epoch [2/15], Training Loss: 0.8864, Validation Loss: 1.3929
Epoch [3/15], Training Loss: 0.6124, Validation Loss: 0.9090
Epoch [4/15], Training Loss: 0.7416, Validation Loss: 1.5024
Epoch [5/15], Training Loss: 0.5884, Validation Loss: 0.9160
Epoch [6/15], Training Loss: 0.6508, Validation Loss: 0.9585
Epoch [7/15], Training Loss: 0.2924, Validation Loss: 1.3900
End of training due to too many fluctuation
Result:
{'Macro Precision': 0.6244, 'Macro Recall': 0.617, 'Macro F1-Score': 0.6096, 'Micro Precision': 0.618, 'Micro Recall': 0.618, 'Micro F1-Score': 0.618, 'Accuracy': 0.618}
Fold 1:
{'Macro Precision': 0.6282, 'Macro Recall': 0.5974, 'Macro F1-Score': 0.6077, 'Micro Precision': 0.5855, 'Micro Recall': 0.5855, 'Micro F1-Score': 0.5855, 'Accuracy': 0.5855}

Fold 2:
{'Macro Precision': 0.6066, 'Macro Recall': 0.5863, 'Macro F1-Score': 0.5856, 'Micro Precision': 0.5684, 'Micro Recall': 0.5684, 'Micro F1-Score': 0.5684, 'Accuracy': 0.5684}

Fold 3:
{'Macro Precision': 0.6008, 'Macro Recall': 0.6001, 'Macro F1-Score': 0.6001, 'Micro Precision': 0.5983, 'Micro Recall': 0.5983, 'Micro F1-Score': 0.5983, 'Accuracy': 0.5983}

Fold 4:
{'Macro Precision': 0.5721, 'Macro Recall': 0.569, 'Macro F1-Score': 0.5651, 'Micro Precision': 0.547, 'Micro Recall': 0.547, 'Micro F1-Score': 0.547, 'Accuracy': 0.547}

Fold 5:
{'Macro Precision': 0.6051, 'Macro Recall': 0.6002, 'Macro F1-Score': 0.5955, 'Micro Precision': 0.5769, 'Micro Recall': 0.5769, 'Micro F1-Score': 0.5769, 'Accuracy': 0.5769}

Fold 6:
{'Macro Precision': 0.6219, 'Macro Recall': 0.6203, 'Macro F1-Score': 0.6175, 'Micro Precision': 0.6309, 'Micro Recall': 0.6309, 'Micro F1-Score': 0.6309, 'Accuracy': 0.6309}

Fold 7:
{'Macro Precision': 0.5601, 'Macro Recall': 0.5565, 'Macro F1-Score': 0.5398, 'Micro Precision': 0.5408, 'Micro Recall': 0.5408, 'Micro F1-Score': 0.5408, 'Accuracy': 0.5408}

Fold 8:
{'Macro Precision': 0.5277, 'Macro Recall': 0.5086, 'Macro F1-Score': 0.4835, 'Micro Precision': 0.4979, 'Micro Recall': 0.4979, 'Micro F1-Score': 0.4979, 'Accuracy': 0.4979}

Fold 9:
{'Macro Precision': 0.6585, 'Macro Recall': 0.5941, 'Macro F1-Score': 0.579, 'Micro Precision': 0.5579, 'Micro Recall': 0.5579, 'Micro F1-Score': 0.5579, 'Accuracy': 0.5579}

Fold 10:
{'Macro Precision': 0.6244, 'Macro Recall': 0.617, 'Macro F1-Score': 0.6096, 'Micro Precision': 0.618, 'Micro Recall': 0.618, 'Micro F1-Score': 0.618, 'Accuracy': 0.618}

Averages:
{'Macro Precision': 0.6005, 'Macro Recall': 0.585, 'Macro F1-Score': 0.5783, 'Micro Precision': 0.5722, 'Micro Recall': 0.5722, 'Micro F1-Score': 0.5722, 'Accuracy': 0.5722}
-----------------------------
"""