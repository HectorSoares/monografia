import numpy as np 
import cv2
from collections import Counter


def removeNoise(img):
    return cv2.medianBlur(img, 5)

def returnSkewAngle(max):
    angle = cv2.minAreaRect(max)[-1]
    if angle < -45:
        angle = 90 + angle
    return angle
    

def rotateImage(img, angle):
    height, width, _ = img.shape
    center = (width/2, height/2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def filterHorizontalRectangles(contours):
    filtered_contours = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Obtém o retângulo delimitador do contorno
        aspect_ratio = float(w) / h  # Calcula a relação altura/largura
        
        # Define um limite para considerar o contorno como um retângulo horizontal
        if aspect_ratio > 1.5:
            filtered_contours.append(contour)
    
    return filtered_contours

def removeSkew(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    horizontal_rectangles = filterHorizontalRectangles(contours)
    
    horizontal_rectangles = sorted(horizontal_rectangles, key=cv2.contourArea, reverse=True)
    
    max_cnt = horizontal_rectangles[0]
    
    angle = returnSkewAngle(max_cnt)
    return rotateImage(img, angle)

