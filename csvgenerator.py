from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, img_as_float, color
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from IPython import display
# Ignore warnings
import warnings
import csv
import copy
from itertools import cycle
from PIL import Image
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

def get_csv_path():
    return os.path.join(os.getcwd(), 'image_files.csv')

def generate_csv():
    cwd = os.getcwd()  
    csvFilePath = os.path.join(cwd, 'image_files.csv')
    surPath = os.path.join(cwd, '../Patches', 'survival')
    decPath = os.path.join(cwd, '../Patches', 'deceased')
    fileListSur = [name for name in os.listdir(surPath) if 
                      os.path.isfile(os.path.join(surPath, name))]
    fileListDec = [name for name in os.listdir(decPath) if 
                      os.path.isfile(os.path.join(decPath, name))]  
    for index, item in enumerate(fileListSur):
        if index%100000 == 0:
            print(str(index/len(fileListSur)*100)+"%")
        try:
            img = Image.open(os.path.join(surPath, item)) # open the image file
            img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            fileListSur.pop(index)
            print('Bad file:', item) # print out the names of corrupt files
        
    for index, item in enumerate(fileListDec):
        if index%100000 == 0:
            print(str(index/len(fileListDec)*100)+"%")
        try:
            img = Image.open(os.path.join(decPath, item)) # open the image file
            img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            fileListDec.pop(index)
            print('Bad file:', item) # print out the names of corrupt files
         
    fileListSur.sort()
    fileListDec.sort()
    
    numOfSur = len(fileListSur)
    numOfDec = len(fileListDec)
    numOfPatch = max(numOfSur, numOfDec)
    zipList = zip(cycle(fileListSur), fileListDec)
    zipList = list(zipList)
    print(numOfPatch)
    print(zipList[0])
    
    with open(csvFilePath, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', 
                                quoting=csv.QUOTE_MINIMAL)
        for i in range(numOfPatch):
            
            sur = os.path.join(surPath, str(zipList[i][0]))
            dec = os.path.join(decPath, str(zipList[i][1]))
            filewriter.writerow([sur, dec])
