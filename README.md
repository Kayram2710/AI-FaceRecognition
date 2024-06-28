# AI-FaceRecognition

## Members

| Name | Student ID |
| --- | --- |
| Karim El Assaad | 40127808 |
| Asif Khan | 40211000 |
| Hazem Mohamed | 40184419 |

## Components
### BiasEvaluator.py
> This python script sets up a runBiasEvaluation function to evaluate an assigned model based on biases it might contain

> The result of the analysis will be printed on screen

> In order to execute this script it requried a saved model that fits any of our CNN structures, the .pth file must be stored in a local folder called "SavedModels" located where this script will be executed, it is required to have a "Cleaned Dataset" folder, such as the one in this repository and a fileld up "DataAnnotations.csv" present in the directory where the script will be executed, this file can be generated after running "BiasDataSetup.py". It also requires the dependency torch, torchvision, and sklearn installed on the system

> The execution can be customized by changing the calls to the runBiasEvaluation function at the end of the script, the function will take any parameter as a path, it should be the name of the .pth file, if it finds the model and the csv file it will start

### BiasDataSetup.py
> This python script sets up a series of function to evaluate setup the directory for bias analysis

> The program will prompt the user to classify individuals on their screen. It will generate a "DataAnnotations.csv in the directory

> In order to execute this script it requried to have a "Cleaned Dataset" folder. It also requires the dependency torch, torchvision, tkit, pillow, and sklearn installed on the system

### Cleaned Dataset Folder
> This is the folder populated/produced by DataCleaner.py

### DataAnnotations.csv
> This is the csv file created by the script BiasDataSetup.py

> It is a user generate cross-refrance sheet to retrieve the annotations of a specific image

### DataCleaner.py
> This python script will iterate through a designated folder that has sub-folder that categorizes images, as its iterating through the images, it will "clean" them and save them in the "Cleaned Dataset" Folder

> It will select 600 images from the sub-folders as a hard limit

> The folder used by the script is called "Raw Data" It is a folder established locally that contains the raw images we obtained

> It can be executed by simply calling it from the command line, with the prerequisite of having the dependency pillow installed, additionally two existing folder should be established in the same directory where the script will be executed, one should be called "Raw Data" and the other "Cleaned Dataset", both folders need to have the exact same sub-folders (same names) within them, and the "Raw Data" folder needs to contain images in its sub folder to produce the desired results

### DataVisualer.py
> This python script will iterate through the "Cleaned Dataset" Folder and perform multiple operations using the matplot library

> The results of these operations will be displayed on screen in a pop tab, once one tab is closed the next will appear

> It can be executed by simply calling it from the command line, with the prerequisite of having the dependency matplotlib, numpy, and pillow installed, additionally, it is required to have a "Cleaned Dataset" folder, such as the one in this repository, present in the directory where the script will be executed

### KFoldEvaluation.py
> This python script will perform K-fold cross referance analysis on a model

> The results of these operations will be displayed in real time in the console

> In order to execute this script it requried a saved model that fits any of our CNN structures, the .pth file must be stored in a local folder called "SavedModels" located where this script will be executed, it is required to have a "Cleaned Dataset" folder, such as the one in this repository. It also requires the dependency torch, torchvision, and sklearn installed on the system

> The execution can be customized by changing parameters in the function call runKfold at the end of the script, most notable changing the type will allow the kfold to run on differant CNN structures, 0(default) for MainModel(MainModel.py) 1 for our first varient (ModelVarient1.py) and 2 for our second varient (ModelVarient2.py), or the number of folds and patience level of the trainer

### Models Scripts Folder
> This folder contains three python file, each of them contains a differantly structured CNN

MainModel.py:
> This script contains our main model, it has two convolutional layer and a kernel size of 3x3

> It is not made to be executed on its own and is instead ran in the ModelTrainer.py script

> It will result in a trained model with its structured being saved in a local directory

ModelVarient1.py:
> This script contains our first varient model, it has three convolutional layer and a kernel size of 5x5

> The idea behind these variations is too check to see if a higher level of complextion in the model could lead to better results 

> It is not made to be executed on its own and is instead ran in the ModelTrainer.py script

> It will result in a trained model with its structured being saved in a local directory

ModelVarient2.py:
> This script contains our second varient model, it has one convolutional layer and a kernel size of 1x1

> The idea behind these variations is too check to see if a lower level of complextion in the model does indeed lead to significantly worst results 

> It is not made to be executed on its own and is instead ran in the ModelTrainer.py script

> It will result in a trained model with its structured being saved in a local directory

### ModelEvaluator.py
> This python script sets up an evaluate function that will calcualte many mathematical properties regarding a saved and trained AI model from the local machine

> The results of the function are printed on screen once the script is ran and the confusion matrices pop up in a tab as its being ran

> It can be executed by simply calling it from the command line, with the prerequisite of having the dependency torch, torchvision, and sklearn installed, additionally, it is required to have a "SavedModels" folder, present in the directory where the script will be executed, this folder needs to have the three models that will be generated after running ModelTrainer.py present

> The execution can be customized by removing some of the calls to the evaluate function at the end of the script, it will take any path as a parameter and run as long as it finds the .pth file in the "SavedModels" folder

### ModelTrainer.py
> This python script sets up a train function that will dynamically train multiple AI model over the course of maximum 15 epoches and save the resulting model in a file

> The real time progress of the training is printed on screen as the models are being trained and a file is saved or overwritten in the local "SavedModels" folder

> It can be executed by simply calling it from the command line, with the prerequisite of having the dependency torch, torchvision, and sklearn installed, additionally, it is required to have a "Cleaned Dataset" folder, such as the one in this repository and a "SavedModels" folder, present in the directory where the script will be executed

> The execution can be customized by changing the calls to the train function at the end of the script, the function will take any parameter as a path, you can add paramaeters such as "type" that will train the model under differant structures, 0(default) for MainModel(MainModel.py) 1 for our first varient (ModelVarient1.py) and 2 for our second varient (ModelVarient2.py)

### Visualisation Results Folder
> This folder contains the results saved while executing DataVisualer.py

> Additionally it contains the confusion matrices generated while executing ModelEvaluator.py

> It is also subdivided into two folders, one that contains visualasition before we adjusted our project for bias mitigation and one for after 

### WebPageDownloader.py
> This is the python script that what used to collect raw data for the focused/engaged category

> It is used by changing the link and assigning a starting image ID so that the script can automate data collection from a site that does not allow large batch downloads automatically.

> The values were  manually inputed as the process still needed user supervision

> It can be executed by simply calling it from the command line, with the prerequisite of having the dependency selenium and webdriver_manager installed

## Dataset Sources:
### Source of Raw Data for Happy, Angry, Neutral
https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data

### Source of Raw Data for Focused
https://www.freepik.com/search?ai=excluded&format=search&last_filter=type&last_value=photo&people=include&people_range=1&query=focused+people&selection=1&type=photo
