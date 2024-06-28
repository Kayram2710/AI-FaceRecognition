import torch
import torch.nn as network
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold, train_test_split
from ModelsScript.ModelMain import FaceRecognitionModel as Main
from ModelsScript.ModelVarient1 import FaceRecognitionModel as V1
from ModelsScript.ModelVarient2 import FaceRecognitionModel as V2

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
def train(model, trainingData, validationData, type=0, epochs = 15, treshold = 2):
    #Writing parameters for early stoping procedure
    fluctuationTreshold = treshold #Patience level of 2
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
def runKfold(folds=10, type=0, epochs = 15, treshold = 2):

    #Setting up Kfold
    kf = KFold(n_splits=folds, shuffle=True, random_state=50)
    results = {}

    #Retrive dasetIndices
    datasetIndices = getDatasetIndex()

    #Start enumerated loop for kfold
    for fold, (trainIndices, testIndices) in enumerate(kf.split(datasetIndices)):

        #Instatiate Model
        #Instantiate Model Depending on type fed
        if(type == 1):
            model = V1().to(device)
        elif(type == 2):
            model = V2().to(device)
        else:
            model = Main().to(device) #Allows toss up path

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
        model = train(model, trainingData, validationData,type,epochs,treshold)

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

#runKfold(folds=10 ,type=0, treshold=2) #Kfold to match old model (pre-bias Mitigation)

runKfold(folds=10 ,type=1, treshold=3) #Kfold to match new model

"""
--------Log Output Pre-Bias Mitigation-----------
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
--------Log Output Post Mitigation for Variant 1-----------
Epoch [1/15], Training Loss: 1.4501, Validation Loss: 1.1627
Epoch [2/15], Training Loss: 0.7838, Validation Loss: 1.0487
Epoch [3/15], Training Loss: 1.1335, Validation Loss: 0.9497
Epoch [4/15], Training Loss: 0.9697, Validation Loss: 1.0921
Epoch [5/15], Training Loss: 1.0606, Validation Loss: 0.8692
Epoch [6/15], Training Loss: 0.7835, Validation Loss: 0.8005
Epoch [7/15], Training Loss: 0.6033, Validation Loss: 1.2382
Epoch [8/15], Training Loss: 0.3184, Validation Loss: 1.4174
Epoch [9/15], Training Loss: 0.1839, Validation Loss: 1.3122
Epoch [10/15], Training Loss: 0.1037, Validation Loss: 1.4122
Epoch [11/15], Training Loss: 0.0685, Validation Loss: 2.8086
Epoch [12/15], Training Loss: 0.2423, Validation Loss: 2.8332
End of training due to too many fluctuation
Result:
{'Macro Precision': 0.4903, 'Macro Recall': 0.5001, 'Macro F1-Score': 0.4775, 'Micro Precision': 0.4867, 'Micro Recall': 0.4867, 'Micro F1-Score': 0.4867, 'Accuracy': 0.4867}
Epoch [1/15], Training Loss: 0.9621, Validation Loss: 1.1008
Epoch [2/15], Training Loss: 0.7809, Validation Loss: 0.8876
Epoch [3/15], Training Loss: 0.9188, Validation Loss: 0.9704
Epoch [4/15], Training Loss: 0.8286, Validation Loss: 0.9708
Epoch [5/15], Training Loss: 1.2379, Validation Loss: 0.9040
Epoch [6/15], Training Loss: 0.9447, Validation Loss: 0.8675
Epoch [7/15], Training Loss: 0.0936, Validation Loss: 1.4223
Epoch [8/15], Training Loss: 0.0244, Validation Loss: 1.1303
Epoch [9/15], Training Loss: 0.0704, Validation Loss: 2.7175
Epoch [10/15], Training Loss: 0.0097, Validation Loss: 1.3240
Epoch [11/15], Training Loss: 0.0028, Validation Loss: 2.6772
Epoch [12/15], Training Loss: 0.0004, Validation Loss: 2.7093
Epoch [13/15], Training Loss: 0.0064, Validation Loss: 2.2554
Epoch [14/15], Training Loss: 0.0013, Validation Loss: 2.6722
Epoch [15/15], Training Loss: 0.0002, Validation Loss: 2.5894
Result:
{'Macro Precision': 0.5311, 'Macro Recall': 0.5423, 'Macro F1-Score': 0.5357, 'Micro Precision': 0.5209, 'Micro Recall': 0.5209, 'Micro F1-Score': 0.5209, 'Accuracy': 0.5209}
Epoch [1/15], Training Loss: 1.0224, Validation Loss: 1.1420
Epoch [2/15], Training Loss: 1.3521, Validation Loss: 1.1098
Epoch [3/15], Training Loss: 0.7865, Validation Loss: 1.0619
Epoch [4/15], Training Loss: 0.6628, Validation Loss: 0.8148
Epoch [5/15], Training Loss: 0.1020, Validation Loss: 1.5162
Epoch [6/15], Training Loss: 0.5773, Validation Loss: 2.3477
Epoch [7/15], Training Loss: 0.0144, Validation Loss: 1.6427
Epoch [8/15], Training Loss: 0.0290, Validation Loss: 2.1918
Epoch [9/15], Training Loss: 0.0022, Validation Loss: 3.0009
Epoch [10/15], Training Loss: 0.0026, Validation Loss: 2.1042
Epoch [11/15], Training Loss: 0.6407, Validation Loss: 3.1773
Epoch [12/15], Training Loss: 0.0123, Validation Loss: 3.7873
Epoch [13/15], Training Loss: 0.0007, Validation Loss: 2.4586
Epoch [14/15], Training Loss: 0.0013, Validation Loss: 2.8613
Epoch [15/15], Training Loss: 0.0041, Validation Loss: 2.1772
Result:
{'Macro Precision': 0.5915, 'Macro Recall': 0.6025, 'Macro F1-Score': 0.5959, 'Micro Precision': 0.597, 'Micro Recall': 0.597, 'Micro F1-Score': 0.597, 'Accuracy': 0.597}   
Epoch [1/15], Training Loss: 0.9594, Validation Loss: 1.0507
Epoch [2/15], Training Loss: 0.8965, Validation Loss: 1.0405
Epoch [3/15], Training Loss: 1.4305, Validation Loss: 0.9356
Epoch [4/15], Training Loss: 0.6325, Validation Loss: 0.9834
Epoch [5/15], Training Loss: 0.5561, Validation Loss: 1.8453
Epoch [6/15], Training Loss: 0.4080, Validation Loss: 1.6083
Epoch [7/15], Training Loss: 0.0035, Validation Loss: 1.7126
Epoch [8/15], Training Loss: 0.0702, Validation Loss: 1.9323
Epoch [9/15], Training Loss: 0.0016, Validation Loss: 1.0107
Epoch [10/15], Training Loss: 0.0027, Validation Loss: 3.2418
Epoch [11/15], Training Loss: 0.0233, Validation Loss: 2.7883
Epoch [12/15], Training Loss: 0.0009, Validation Loss: 2.9416
Epoch [13/15], Training Loss: 0.0024, Validation Loss: 2.4332
Epoch [14/15], Training Loss: 0.0007, Validation Loss: 1.7113
Epoch [15/15], Training Loss: 0.0014, Validation Loss: 2.9388
Result:
{'Macro Precision': 0.5114, 'Macro Recall': 0.5163, 'Macro F1-Score': 0.5131, 'Micro Precision': 0.4924, 'Micro Recall': 0.4924, 'Micro F1-Score': 0.4924, 'Accuracy': 0.4924}
Epoch [1/15], Training Loss: 1.1206, Validation Loss: 1.1518
Epoch [2/15], Training Loss: 0.8515, Validation Loss: 1.0138
Epoch [3/15], Training Loss: 1.4466, Validation Loss: 1.1733
Epoch [4/15], Training Loss: 0.8636, Validation Loss: 1.0035
Epoch [5/15], Training Loss: 1.1979, Validation Loss: 1.0629
Epoch [6/15], Training Loss: 0.4096, Validation Loss: 1.0348
Epoch [7/15], Training Loss: 0.2174, Validation Loss: 2.8563
Epoch [8/15], Training Loss: 0.0113, Validation Loss: 1.4776
Epoch [9/15], Training Loss: 0.0326, Validation Loss: 2.3018
Epoch [10/15], Training Loss: 0.0038, Validation Loss: 1.4126
Epoch [11/15], Training Loss: 0.0022, Validation Loss: 1.5126
Epoch [12/15], Training Loss: 0.0065, Validation Loss: 2.2356
Epoch [13/15], Training Loss: 0.0058, Validation Loss: 1.8866
Epoch [14/15], Training Loss: 0.0003, Validation Loss: 3.6192
Epoch [15/15], Training Loss: 0.0040, Validation Loss: 4.9784
Result:
{'Macro Precision': 0.5793, 'Macro Recall': 0.5858, 'Macro F1-Score': 0.5818, 'Micro Precision': 0.5954, 'Micro Recall': 0.5954, 'Micro F1-Score': 0.5954, 'Accuracy': 0.5954}
Epoch [1/15], Training Loss: 1.3664, Validation Loss: 1.0856
Epoch [2/15], Training Loss: 0.9216, Validation Loss: 1.0145
Epoch [3/15], Training Loss: 0.8730, Validation Loss: 1.0548
Epoch [4/15], Training Loss: 0.5185, Validation Loss: 0.9201
Epoch [5/15], Training Loss: 0.1750, Validation Loss: 0.9681
Epoch [6/15], Training Loss: 0.0898, Validation Loss: 1.5325
Epoch [7/15], Training Loss: 0.0142, Validation Loss: 1.0487
Epoch [8/15], Training Loss: 0.0101, Validation Loss: 1.2847
Epoch [9/15], Training Loss: 0.0005, Validation Loss: 2.6448
Epoch [10/15], Training Loss: 0.0014, Validation Loss: 2.5488
Epoch [11/15], Training Loss: 0.0010, Validation Loss: 2.4898
Epoch [12/15], Training Loss: 0.0007, Validation Loss: 1.8826
Epoch [13/15], Training Loss: 0.0032, Validation Loss: 2.5147
Epoch [14/15], Training Loss: 0.0019, Validation Loss: 2.0673
Epoch [15/15], Training Loss: 0.0021, Validation Loss: 2.5798
Result:
{'Macro Precision': 0.5352, 'Macro Recall': 0.5449, 'Macro F1-Score': 0.5396, 'Micro Precision': 0.5229, 'Micro Recall': 0.5229, 'Micro F1-Score': 0.5229, 'Accuracy': 0.5229}
Epoch [1/15], Training Loss: 0.9737, Validation Loss: 1.1560
Epoch [2/15], Training Loss: 1.0829, Validation Loss: 1.1117
Epoch [3/15], Training Loss: 1.0302, Validation Loss: 1.3157
Epoch [4/15], Training Loss: 1.2364, Validation Loss: 1.0076
Epoch [5/15], Training Loss: 1.3583, Validation Loss: 1.1551
Epoch [6/15], Training Loss: 0.9233, Validation Loss: 1.2084
Epoch [7/15], Training Loss: 1.1721, Validation Loss: 0.8621
Epoch [8/15], Training Loss: 1.1259, Validation Loss: 1.1584
Epoch [9/15], Training Loss: 0.2905, Validation Loss: 1.0505
Epoch [10/15], Training Loss: 0.4257, Validation Loss: 2.3390
Epoch [11/15], Training Loss: 0.2124, Validation Loss: 2.1299
Epoch [12/15], Training Loss: 0.0071, Validation Loss: 3.1754
Epoch [13/15], Training Loss: 0.1767, Validation Loss: 2.1384
Epoch [14/15], Training Loss: 0.0031, Validation Loss: 2.7361
Epoch [15/15], Training Loss: 0.1155, Validation Loss: 2.2161
Result:
{'Macro Precision': 0.4953, 'Macro Recall': 0.5151, 'Macro F1-Score': 0.4971, 'Micro Precision': 0.5115, 'Micro Recall': 0.5115, 'Micro F1-Score': 0.5115, 'Accuracy': 0.5115}
Epoch [1/15], Training Loss: 1.7201, Validation Loss: 1.2732
Epoch [2/15], Training Loss: 0.9815, Validation Loss: 0.9678
Epoch [3/15], Training Loss: 1.0768, Validation Loss: 1.1553
Epoch [4/15], Training Loss: 1.4787, Validation Loss: 1.1930
Epoch [5/15], Training Loss: 0.8595, Validation Loss: 0.8998
Epoch [6/15], Training Loss: 0.3818, Validation Loss: 0.9960
Epoch [7/15], Training Loss: 0.8155, Validation Loss: 1.2707
Epoch [8/15], Training Loss: 0.5421, Validation Loss: 1.0741
Epoch [9/15], Training Loss: 0.3963, Validation Loss: 1.6897
Epoch [10/15], Training Loss: 0.0732, Validation Loss: 2.4886
Epoch [11/15], Training Loss: 0.0809, Validation Loss: 2.9681
End of training due to too many fluctuation
Result:
{'Macro Precision': 0.5205, 'Macro Recall': 0.5301, 'Macro F1-Score': 0.5245, 'Micro Precision': 0.5191, 'Micro Recall': 0.5191, 'Micro F1-Score': 0.5191, 'Accuracy': 0.5191}
Epoch [1/15], Training Loss: 1.1807, Validation Loss: 1.3629
Epoch [2/15], Training Loss: 1.6288, Validation Loss: 1.0636
Epoch [3/15], Training Loss: 0.9131, Validation Loss: 0.9541
Epoch [4/15], Training Loss: 0.8397, Validation Loss: 1.0277
Epoch [5/15], Training Loss: 1.1447, Validation Loss: 1.1613
Epoch [6/15], Training Loss: 0.9196, Validation Loss: 1.1084
Epoch [7/15], Training Loss: 0.3821, Validation Loss: 1.2565
Epoch [8/15], Training Loss: 0.3311, Validation Loss: 1.2483
Epoch [9/15], Training Loss: 0.0270, Validation Loss: 1.6063
Epoch [10/15], Training Loss: 0.0860, Validation Loss: 1.1786
Epoch [11/15], Training Loss: 0.0619, Validation Loss: 3.7836
Epoch [12/15], Training Loss: 0.3600, Validation Loss: 2.0012
Epoch [13/15], Training Loss: 0.0090, Validation Loss: 2.8290
Epoch [14/15], Training Loss: 0.0043, Validation Loss: 3.4035
Epoch [15/15], Training Loss: 0.0011, Validation Loss: 2.7072
Result:
{'Macro Precision': 0.5154, 'Macro Recall': 0.5302, 'Macro F1-Score': 0.5201, 'Micro Precision': 0.5191, 'Micro Recall': 0.5191, 'Micro F1-Score': 0.5191, 'Accuracy': 0.5191}
Epoch [1/15], Training Loss: 1.2259, Validation Loss: 1.0275
Epoch [2/15], Training Loss: 1.3875, Validation Loss: 1.0125
Epoch [3/15], Training Loss: 1.0542, Validation Loss: 1.0253
Epoch [4/15], Training Loss: 0.3837, Validation Loss: 0.8661
Epoch [5/15], Training Loss: 1.1800, Validation Loss: 1.0645
Epoch [6/15], Training Loss: 0.5704, Validation Loss: 1.0866
Epoch [7/15], Training Loss: 0.6001, Validation Loss: 1.1240
End of training due to too many fluctuation
Result:
{'Macro Precision': 0.5281, 'Macro Recall': 0.5412, 'Macro F1-Score': 0.5312, 'Micro Precision': 0.5305, 'Micro Recall': 0.5305, 'Micro F1-Score': 0.5305, 'Accuracy': 0.5305}
Fold 1:
{'Macro Precision': 0.4903, 'Macro Recall': 0.5001, 'Macro F1-Score': 0.4775, 'Micro Precision': 0.4867, 'Micro Recall': 0.4867, 'Micro F1-Score': 0.4867, 'Accuracy': 0.4867}

Fold 2:
{'Macro Precision': 0.5311, 'Macro Recall': 0.5423, 'Macro F1-Score': 0.5357, 'Micro Precision': 0.5209, 'Micro Recall': 0.5209, 'Micro F1-Score': 0.5209, 'Accuracy': 0.5209}

Fold 3:
{'Macro Precision': 0.5915, 'Macro Recall': 0.6025, 'Macro F1-Score': 0.5959, 'Micro Precision': 0.597, 'Micro Recall': 0.597, 'Micro F1-Score': 0.597, 'Accuracy': 0.597}   

Fold 4:
{'Macro Precision': 0.5114, 'Macro Recall': 0.5163, 'Macro F1-Score': 0.5131, 'Micro Precision': 0.4924, 'Micro Recall': 0.4924, 'Micro F1-Score': 0.4924, 'Accuracy': 0.4924}

Fold 5:
{'Macro Precision': 0.5793, 'Macro Recall': 0.5858, 'Macro F1-Score': 0.5818, 'Micro Precision': 0.5954, 'Micro Recall': 0.5954, 'Micro F1-Score': 0.5954, 'Accuracy': 0.5954}

Fold 6:
{'Macro Precision': 0.5352, 'Macro Recall': 0.5449, 'Macro F1-Score': 0.5396, 'Micro Precision': 0.5229, 'Micro Recall': 0.5229, 'Micro F1-Score': 0.5229, 'Accuracy': 0.5229}

Fold 7:
{'Macro Precision': 0.4953, 'Macro Recall': 0.5151, 'Macro F1-Score': 0.4971, 'Micro Precision': 0.5115, 'Micro Recall': 0.5115, 'Micro F1-Score': 0.5115, 'Accuracy': 0.5115}

Fold 8:
{'Macro Precision': 0.5205, 'Macro Recall': 0.5301, 'Macro F1-Score': 0.5245, 'Micro Precision': 0.5191, 'Micro Recall': 0.5191, 'Micro F1-Score': 0.5191, 'Accuracy': 0.5191}

Fold 9:
{'Macro Precision': 0.5154, 'Macro Recall': 0.5302, 'Macro F1-Score': 0.5201, 'Micro Precision': 0.5191, 'Micro Recall': 0.5191, 'Micro F1-Score': 0.5191, 'Accuracy': 0.5191}

Fold 10:
{'Macro Precision': 0.5281, 'Macro Recall': 0.5412, 'Macro F1-Score': 0.5312, 'Micro Precision': 0.5305, 'Micro Recall': 0.5305, 'Micro F1-Score': 0.5305, 'Accuracy': 0.5305}

Averages:
{'Macro Precision': 0.5298, 'Macro Recall': 0.5408, 'Macro F1-Score': 0.5317, 'Micro Precision': 0.5296, 'Micro Recall': 0.5296, 'Micro F1-Score': 0.5296, 'Accuracy': 0.5295}
-----------------------------

"""