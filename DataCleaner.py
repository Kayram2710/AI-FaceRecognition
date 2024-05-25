import os
from PIL import Image

rawPath = "Raw Dataset"
destinationPath = "Cleaned Dataset"
size=(256, 256)

#Iterate over files in directory
for category in os.listdir(rawPath):
    source = os.path.join(rawPath, category)
    destination = os.path.join(destinationPath, category)

    for imageFile in os.listdir(source):
        #Open the image to edit
        image = Image.open(os.path.join(source, imageFile))

        #Edit the image
        image = image.resize(size, Image.Resampling.LANCZOS) #Resize image
        image = image.convert('L')  #Convert to grayscale

        #Save the processed image
        image.save(os.path.join(destination, imageFile))

