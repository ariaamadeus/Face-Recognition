from PIL import Image
import os, sys
import cv2
import numpy as np

'''
Converts all images in a directory to '.npy' format.
Use np.save and np.load to save and load the images.
Use it for training your neural networks in ML/DL projects. 
'''

# Path to image directory
# path = "assets/"
# dirs = os.listdir( path )
# dirs.sort()
# print(dirs)
# x_train=[]

# def load_dataset():
#     # Append images to a list
#     for item in dirs:
#         print(path+item)
#         if os.path.isfile(path+item):
#             im = Image.open(path+item).convert("RGB")
#             im = np.array(im)
#             print(im)
#             x_train.append(im)

if __name__ == "__main__":
    
    # load_dataset()
    
    # Convert and save the list of images in '.npy' format
    path = "assets/logo.jpg"
    im = Image.open(path).convert("RGB")
    im = np.array(im)
    imgset=np.array(im)
    np.save("imgds.npy",imgset)