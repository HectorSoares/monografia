import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from cnrr import best_path

def returnRatioToReescale(img):
    height, width, _ = img.shape
    print("original shape: ", height, width)
    new_height = (height//32)*32
    new_width = (width//32)*32
    heigth_ratio = height/new_height
    width_ratio = width/new_width
    print("new shape: ", new_height, new_width,)
    print("ratio: ", heigth_ratio, width_ratio)
    return new_height, new_width, heigth_ratio, width_ratio



def prepareImage(img):
    model = cv2.dnn.readNet("C:\\Repositorios\\tcc\\ocr\\reconhecimento-documentos\\files\\frozen_east_text_detection.pb")
    crnnModel = cv2.dnn.readNet('C:\\Repositorios\\tcc\\ocr\\reconhecimento-documentos\\files\\crnn.onnx')
    new_height, new_width, heigth_ratio, width_ratio = returnRatioToReescale(img)

    blob = cv2.dnn.blobFromImage(img, 1, (new_width, new_height),(123.68, 116.78, 103.94), True, False)

    model.setInput(blob)    
    print(np.array(model.getUnconnectedOutLayersNames()))
    (geometry, scores) = model.forward(model.getUnconnectedOutLayersNames())

    rectangles = []
    confidence_score = []

    for i in range(geometry.shape[2]):
        for j in range(0, geometry.shape[3]):

            if scores[0][0][i][j] < 0.1:
                continue

            bottom_x = int(j*4 + geometry[0][1][i][j])
            bottom_y = int(i*4 + geometry[0][2][i][j])

            top_x = int(j*4 - geometry[0][3][i][j])
            top_y = int(i*4 - geometry[0][0][i][j])

            rectangles.append((top_x, top_y, bottom_x, bottom_y))
            confidence_score.append(float(scores[0][0][i][j]))
    
    fin_boxes = non_max_suppression(np.array(rectangles), probs=confidence_score, overlapThresh=0.5)
    img_copy = img.copy()

    alphabet_set = "0123456789abcdefghijklmnopqrstuvwxyz/ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    blank = '-'

    char_set = blank + alphabet_set

    for(x1,y1,x2,y2) in fin_boxes:
        x1 = int(x1*width_ratio)
        y1 = int(y1*heigth_ratio)
        x2 = int(x2*width_ratio)
        y2 = int(y2*heigth_ratio)

        segment = img[y1:y2, x1:x2, :]
    
        segment_gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
        blob = cv2.dnn.blobFromImage(segment_gray, scalefactor=1/127.5, size=(100,32), mean=127.5)
        
        crnnModel.setInput(blob)
        scores = crnnModel.forward()
        text = best_path(scores, char_set)
        print(text.strip())
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, text.strip(), (x1,y1-2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255),2)
    return img_copy
    

    

