import os
from PIL import Image

#Defined paths
rawPath = "Raw Dataset" #Source for Raw Data, creating String for path
destinationPath = "Cleaned Dataset" #Destination for cleaned Data, creating String for path
size=(256, 256) #Chosen size that pictures will be formatted too

#Creating a value for total amount of pictures desired to be taken from each folder
total = 500

#Iterating over all sub directories in main directory (Raw Data Folder)
for category in os.listdir(rawPath):
    source = os.path.join(rawPath, category) #Creating String source as path for current sub directory
    destination = os.path.join(destinationPath, category) #Creating String destination as relative destination path for current sub directory

    #Creating an iterator count
    count = 0

    #Looping through every image in current sub category
    for imageFile in os.listdir(source):
        #Open the image to edit (Using PIL)
        image = Image.open(os.path.join(source, imageFile))

        #Edit the image
        image = image.resize(size, Image.Resampling.LANCZOS) #Resize image
        image = image.convert('L')  #Convert to grayscale

        #Save the processed image
        image.save(os.path.join(destination, imageFile))
        count = count + 1

        #Status report of progress
        print(f"Progress for {category}: {((count/total)*100):.2f}% transferred")

        #Break when total is reached
        if(count == total):
            break
