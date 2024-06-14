# AI-FaceRecognition

## Members

| Name | Student ID |
| --- | --- |
| Karim El Assaad | 40127808 |
| Asif Khan | 40211000 |
| Hazem Mohamed | 40184419 |

## Components

### Cleaned Dataset Folder
> This is the folder populated/produced by DataCleaner.py

### DataCleaner.py
> This python script will iterate through a designated folder that has sub-folder that categorizes images, as its iterating through the images, it will "clean" them and save them in the "Cleaned Dataset" Folder

> It will select 600 images from the sub-folders as a hard limit

> The folder used by the script is called "Raw Data" It is a folder established locally that contains the raw images we obtained

> It can be executed by simply calling it from the command line, with the prerequisite of having the dependency pillow installed, additionally two existing folder should be established in the same directory where the script will be executed, one should be called "Raw Data" and the other "Cleaned Dataset", both folders need to have the exact same sub-folders (same names) within them, and the "Raw Data" folder needs to contain images in its sub folder to produce the desired results

### DataVisualer.py
> This python script will iterate through the "Cleaned Dataset" Folder and perform multiple operations using the matplot library

> The results of these operations will be displayed on screen in a pop tab, once one tab is closed the next will appear

> It can be executed by simply calling it from the command line, with the prerequisite of having the dependency matplotlib, numpy, and pillow installed, additionally, it is required to have a "Cleaned Dataset" folder, such as the one in this repository, present in the directory where the script will be executed

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

> The results of the function are printed on screen once the script is ran

> It can be executed by simply calling it from the command line, with the prerequisite of having the dependency torch, torchvision, and sklearn installed, additionally, it is required to have a "SavedModels" folder, present in the directory where the script will be executed, this folder needs to have the three models that will be generated after running ModelTrainer.py present

> The execution can be customized by removing some of the calls to the evaluate function at the end of the script, but unlike ModelTrainer.py it does not allow to pass a parameter that isnt one of the following: "MainModel", "Varient1" , "Varient2"

### ModelTrainer.py
> This python script sets up a train function that will dynamically train multiple AI model over the course of maximum 15 epoches and save the resulting model in a file

> The real time progress of the training is printed on screen as the models are being trained and a file is saved or overwritten in the local "SavedModels" folder

> It can be executed by simply calling it from the command line, with the prerequisite of having the dependency torch, torchvision, and sklearn installed, additionally, it is required to have a "Cleaned Dataset" folder, such as the one in this repository and a "SavedModels" folder, present in the directory where the script will be executed

> The execution can be customized by changing the calls to the train function at the end of the script, the function will take any parameter as a path and will train all of them under our base model (MainModel.py) except for paths that are specifically named "Varient1" , and "Varient2", these will be trained as our first varient (ModelVarient1.py) and as our second varient (ModelVarient2.py) respectively

### Visualisation Results Folder
> This folder contains the results saved while executing DataVisualer.py

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
