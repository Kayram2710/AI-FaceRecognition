import numpy as np
import csv
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from torchvision import datasets, transforms
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

    return dataset, testingIndices

#Create function to fill csv
def fillCsv():
    #retrieve necessary info from dataset
    dataset, indices = getDataset()

    #Define dictionary, each key will be a row and it's items will be the differant entries in the column
    csvRows = {}

    filePath = "DataAnnotations.csv"

    #Define headers
    csvRows['headers'] = ["Indices", "Path", "Gender", "Race"]
    
    #Loop through all image ids in testing indices
    for i, imageID in enumerate(indices):
        #Retrive path
        path = dataset.imgs[imageID][0]

        #Call the form
        race, gender = form(path)
        print(race, gender) #And print decision

        #Add row to dictinoary
        csvRows[i] = (imageID, path, gender, race)

    #Open csv to be written
    with open(filePath, mode='w', newline='') as file:
        #Define write for file
        writer = csv.writer(file)
        
        #Loop through written dictionary
        for key, value in csvRows.items():
            #Write to row
            writer.writerow(value)


# This function was generated mostly with CHATGPT when prompted with the following:
# make me a function that takes an image path as a parameter, 
# it will display the image to the user in a pop up and next to it in the pop up 
# it will give me the option of filling two forms which are multiple choice,
# once I submit it the form will update two variables in the code and the function will return them.
# It was then heavily modified and optimized
#Creation of form
def form(path):

    #Define hypervariables variables
    selectedRace = None
    selectedGender = None

    #Submit form button
    def submit_form():
        #Call global variables
        nonlocal selectedRace, selectedGender

        #Check for both inputs
        if not race.get() or not gender.get():
            messagebox.showwarning("Input Error", "Please select options for both forms.")
        
        #If all passes
        else:

            #Get selection
            selectedRace = race.get()
            selectedGender = gender.get()

            # Close the window
            root.destroy()

    # Create the main window
    root = tk.Tk()
    root.title("Image and Form")

    # Load and display the image
    img = Image.open(path)
    img = ImageTk.PhotoImage(img)
    
    image_label = tk.Label(root, image=img)
    image_label.grid(row=0, column=0, rowspan=4, columnspan=4, padx=10, pady=10)

    gender = tk.StringVar(value="")  # Initialize the variable
    genderOptions = ["Male","Female","Other"] #Initilize the list of options

    race = tk.StringVar(value="")  # Initialize the variable
    raceOptions = ["White","Black","Asian"] #Initilize the list of options

    # Create form labels and radio buttons
    tk.Label(root, text="Select Gender:").grid(row=5, column=1, padx=5, pady=5)
    for i, option in enumerate(genderOptions):
        tk.Radiobutton(root, text=option, variable=gender, value=option).grid(row=(i+6), column=1, padx=5, sticky="w")

     # Create form labels and radio buttons
    tk.Label(root, text="Select Race:").grid(row=5, column=0, padx=5, pady=5)
    for i, option in enumerate(raceOptions):
        tk.Radiobutton(root, text=option, variable=race, value=option).grid(row=(i+6), column=0, padx=5, sticky="w")

    # Create a submit button
    submit = tk.Button(root, text="Submit", command=lambda: submit_form())
    submit.grid(row=10, column=1, pady=20)

    # Start the GUI event loop
    root.mainloop()

    #Returnm selection
    return selectedRace, selectedGender

fillCsv()


"""
--------Log Output-----------
White Male
White Female
White Male
Asian Female
White Male
Black Male
White Female
Black Male
White Male
White Male
Black Male
White Male
White Female
White Male
White Male
White Male
White Female
White Male
White Male
White Male
White Female
White Male
White Female
White Male
Black Male
White Female
White Female
White Male
Black Male
White Female
White Male
White Male
Black Female
White Female
Black Male
Black Female
White Male
White Male
White Female
White Male
White Female
White Male
Black Male
White Female
Black Male
Black Male
White Female
White Male
Black Female
White Female
Black Female
White Female
White Male
White Male
White Male
White Male
White Female
Asian Female
Black Female
White Female
Asian Female
Asian Female
White Male
White Male
White Male
White Male
White Male
White Male
White Female
Black Female
White Female
White Female
White Female
White Male
White Female
Black Male
White Female
Black Male
White Male
White Male
White Male
White Male
White Male
White Female
White Female
White Female
White Female
Asian Female
White Female
White Female
White Female
White Male
White Female
Black Male
Black Male
White Male
White Female
Black Male
White Male
White Male
Asian Female
Black Female
Asian Female
White Other
Black Female
White Male
Black Male
Asian Male
White Female
Black Female
White Female
White Other
White Other
White Female
White Female
Black Male
White Male
White Female
Black Male
White Female
Asian Female
White Male
White Male
White Male
White Other
White Other
White Male
White Female
Asian Female
Black Male
White Male
White Other
White Male
White Male
White Female
White Female
Asian Female
White Other
Black Female
White Other
Asian Female
Black Male
White Male
White Female
Asian Male
White Male
White Female
White Other
Black Male
White Male
White Female
White Other
White Female
White Female
Black Female
White Male
White Female
White Female
White Other
White Other
White Female
White Male
White Other
White Other
Black Male
Black Female
White Female
White Male
White Male
White Male
White Male
White Male
White Male
Black Female
White Other
White Female
White Other
White Other
Black Male
White Female
White Female
White Female
White Female
White Other
White Male
White Male
White Male
White Male
Black Male
White Female
White Female
Asian Male
White Female
White Male
White Female
White Other
White Female
White Female
White Female
White Male
White Male
Black Male
Black Female
Asian Female
White Male
White Male
Black Male
Black Female
White Female
White Male
Black Male
White Male
White Female
White Male
White Male
Asian Male
Asian Male
White Male
White Male
White Female
Asian Female
White Female
White Male
White Male
White Male
White Female
White Female
White Male
Asian Female
White Male
White Male
Asian Female
White Female
White Female
White Other
White Female
White Male
White Male
White Male
White Female
White Female
Asian Female
Black Other
White Female
Black Male
Black Other
White Other
White Male
White Male
White Male
Asian Male
White Female
White Male
Black Female
Asian Male
White Female
Black Male
White Female
White Other
Black Male
White Male
Black Male
White Female
White Male
White Male
White Male
Asian Male
Black Male
White Male
Black Male
White Other
Asian Male
White Male
White Female
White Male
White Male
Asian Male
White Male
White Male
Asian Female
Asian Male
White Female
Asian Male
White Male
White Male
White Male
Black Male
Asian Male
White Male
Black Female
White Female
White Male
White Male
Black Female
Black Other
Asian Female
White Male
White Male
White Male
White Male
White Female
Asian Female
Black Other
White Female
Black Male
White Male
Black Male
White Female
White Male
White Male
White Female
White Male
White Female
White Female
Asian Female
White Other
White Male
Asian Male
White Other
White Other
White Male
White Male
White Female
White Male
White Other
White Male
White Female
White Female
White Male
White Other
Black Male
White Female
White Male
Black Male
White Male
White Male
Asian Other
Asian Other
White Other
White Male
White Male
Black Female
Black Other
White Male
White Male
Asian Male
White Female
White Female
Black Female
White Female
White Other
-----------------------------
"""
