import cv2
from matplotlib import pyplot as plt

def loadImage():
    img = cv2.imread("./imgs/frente.jpg", 0)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def showImage(img):    
    plt.imshow(img)
    plt.show()