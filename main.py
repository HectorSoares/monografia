import numpy as np 
from preProcessing import removeNoise, removeSkew
from east import prepareImage
from util import loadImage, showImage



def preProcessesImage(img):
   
    img = removeNoise(img)
    showImage(img)
    img = removeSkew(img)
    showImage(img)
    return img

def main():
    obj_img = loadImage()
    showImage(obj_img)
    obj_img = preProcessesImage(obj_img)
    showImage(obj_img)
    obj_img = prepareImage(obj_img)
    showImage(obj_img)
    

main()