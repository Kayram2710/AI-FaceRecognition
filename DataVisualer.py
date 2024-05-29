import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

#Declare the dataset source
source = 'Cleaned Dataset'

#Establish a dict that will contain all the paths to the individual images in each respective subfolder
categoriesContent = {}
#Loop through each sub-directory
for category in os.listdir(source):
    #Create path to sub-sirectory
    categoryPath = os.path.join(source, category)
    
    #Create a list to store result for images
    images = []
    #Loop through each image in sub-directory
    for img in os.listdir(categoryPath):
        images.append(os.path.join(categoryPath, img)) #Append image path to list
    
    #Add to dictionary in respective category
    categoriesContent[category] = images

#Define function to plot distribution of images across differant categories
def PlotNumericalDistribution():
    #Create empty lists
    categoryCounts = []
    categories = []

    #Fill lists by looping through categoriesContent dictionary
    for key, value in categoriesContent.items():
        categories.append(key)
        categoryCounts.append(len(value))

    #Plotting barchart
    plt.figure(figsize=(10, 5))
    plt.title('Category Distribution')
    plt.xlabel('Categories')
    plt.ylabel('Number of Images')
    plt.bar(categories, categoryCounts, color='green')
    plt.show()

#Creating funtion that will plot the pixel density across all iamges
def PlotPixelDensity():
    #Creating figure
    plt.figure(figsize=(15, 10))

    #Create iterator starting at 1
    count = 1
    #Loop through dictionary
    for key, value in categoriesContent.items():
        #Create numpy array of total pixels to store array of values
        allPixels = np.array([])

        #Loop through all images
        for path in value:
            #Open each iamge and ravel them
            pixels = np.array(Image.open(path)).ravel()
            #Concatenate to the numpy array the new value
            allPixels = np.concatenate([allPixels, pixels])
        
        #Add to the plot results
        plt.subplot(2, 2, count)
        plt.hist(allPixels, bins=256, color='gray', alpha=0.75)
        plt.title(f'Pixel Intensity Distribution for {key}')

        #Increase iterator
        count = count + 1

    #Plot results
    plt.tight_layout()
    plt.show()

#Creating the function to plot 15 picture samples and their pixel histogram for each category
def PlotSamplePictures():
    #Loop through dictionary
    for key, value in categoriesContent.items():
        #Pick 15 random samples
        samples = random.sample(value, min(len(value), 15))

        #Creating figure
        plt.figure(figsize=(10, 20))
        #Create iterator starting at 1
        count = 0
        #Loop through sampled images
        for path in samples:
            
            #Open image
            img = np.array(Image.open(path))
            
            #Plot images
            plt.subplot(5, 6, 2*count+1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            
            #Plot histogram
            plt.subplot(5, 6, 2*count+2)
            plt.hist(img.ravel(), bins=256, color='gray')
            plt.axis('off')

            #Increase Iterator
            count = count + 1

        #Show scene
        plt.suptitle(f'Sample Images and their Histograms for {key}')
        plt.show()

PlotNumericalDistribution()
PlotPixelDensity()
PlotSamplePictures()