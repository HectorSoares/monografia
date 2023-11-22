import numpy as np 
from preProcessing import removeNoise, removeSkew, sharpImage, addContrast, manipular_imagem
from east import prepareImage
from util import loadImage, showImage



def preProcessesImage(img):
   
    #img = removeNoise(img)
    #showImage(img)
    #img = removeSkew(img)
    #showImage(img)     
    img = sharpImage(img)
    #showImage(img)  
    #img = addContrast(img)
    #showImage(img)  
    return img

def main():
    obj_img = loadImage()
    showImage(obj_img)
    obj_img = preProcessesImage(obj_img)
    obj_img = prepareImage(obj_img)
    showImage(obj_img)
    

main()