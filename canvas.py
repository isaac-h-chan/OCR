import cv2
import numpy as np 
import matplotlib.pyplot as plt
import ocrCNN as cnn
import torch
from pathlib import Path


MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'OCR_CNN_MODEL.pth'
SAVE_PATH = MODEL_PATH / MODEL_NAME

# Device agnostic code >> sets 'device' to GPU if nVIDIA GPU is available, otherwise it is set to CPU
device = ''

if Path.exists(SAVE_PATH):
    model0 = cnn.ocrModel(1, 10, len(cnn.train_data.classes))
    model0.load_state_dict(state_dict=torch.load(SAVE_PATH, map_location='cpu'))
    print('Model Loaded')
else:
    model0 = cnn.ocrModel(1, 10, len(cnn.train_data.classes))


drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None

# mouse callback function
def line_drawing(event, x, y, flag, params):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=25)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=25)        


img = np.zeros((512,512), np.float32)
cv2.namedWindow('test draw')
cv2.setMouseCallback('test draw',line_drawing)
count = 0
copy = np.ndarray((28, 28))
while(1):
    cv2.imshow('test draw',img)
    count += 1
    if count == 20:
        count = 0
        copy = cv2.resize(img, (28, 28))/256
        tensor = torch.from_numpy(copy).unsqueeze(dim=0).unsqueeze(dim=0)
        cv2.setWindowTitle("test draw", cnn.predict(tensor, model0))
        
    key = cv2.waitKey(2)  
    if key & 0xFF == 27:
        break
    elif key == ord('c'):
        img = np.zeros((512, 512), np.float32)
cv2.destroyAllWindows()