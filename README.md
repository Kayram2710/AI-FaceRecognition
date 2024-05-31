# AI-FaceRecognition

## Members

| Name | Student ID |
| --- | --- |
| Karim El Assaad | 40127808 |
| Asif Khan | 40211000 |
| Hazem Mohamed | 40184419 |

## Components

### DataCleaner.py
> This python script will iterate through a designated folder that has sub-folder that categorizes images, as its iterating through the images, it will "clean" them and save them in the "Cleaned Dataset" Folder

> It will select 600 images from the sub-folders as a hard limit

> The folder used by the script is called "Raw Data" It is a folder established locally that contains the raw images we obtained

> It can be executed by simply calling it from the command line, with the prerequisite of having the dependency pillow installed, additionally two existing folder should be established in the same directory where the script will be executed, one should be called "Raw Data" and the other "Cleaned Dataset", both folders need to have the exact same sub-folders (same names) within them, and the "Raw Data" folder needs to contain images in its sub folder to produce the desired results

### Cleaned Dataset Folder
> This is the populated by DataCleaner.py

### DataVisualer.py
> This python script will iterate through the "Cleaned Dataset" Folder and perform multiple operations using the matplot library

> The results of these operations will be displayed on screen in a pop tab, once one tab is closed the next will appear

> It can be executed by simply calling it from the command line, with the prerequisite of having the dependency matplotlib, numpy, and pillow installed, additionally, it is required to have a "Cleaned Dataset" folder, such as the one in this repository, present in the directory where the script will be executed

### WebPageDownloader.py
> This python script that what used to collect raw data for the focused/engaged category

> It is used by changing the link and assigning a starting image ID so that the script can automate data collection from a site that does not allow large batch downloads automatically.

> The values were  manually inputed as the process still needed user supervision

> It can be executed by simply calling it from the command line, with the prerequisite of having the dependency selenium and webdriver_manager installed
