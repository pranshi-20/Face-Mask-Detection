from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils
import time
import os
import cv2
import numpy as np
#os.environ['TP_CPP_MIN_LOG_LEVEL']='2'


def detectmask(frame, mask_model):
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces=face_cascade.detectMultiScale(gray, 1.3, 5)
    coordinates=[]
    predictions=[]
    for (x, y, w, h) in faces:
        face=frame[y:y+h, x:x+w]
        face=cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face=cv2.resize(face, (224, 224))
        face=img_to_array(face)
        face=preprocess_input(face)
        face=np.expand_dims(face, axis=0)
        coordinates.append((x, y, w, h))

        if len(face)>0:
            predictions=mask_model.predict(face)
    return coordinates, predictions


model="model_mask1.h5"
mask_model=load_model(model)

cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame=cap.read()
    frame=cv2.flip(frame, 1)

    coordinates, predictions=detectmask(frame, mask_model)
    for r, pred in zip(coordinates, predictions):
        (x, y, w, h)=r
        (mask, without_mask)=pred

        if(mask>=without_mask):
            label="Mask"
            color=(0, 255, 0)

        else:
            label="No Mask"
            color=(0, 0, 255)

        label= "{}: {:.2f}%".format(label, max(mask, without_mask)*100)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.imshow('Frame',frame)
    k=cv2.waitKey(1) & 0xFF
    if k==27:
        cap.release()
        break

cv2.destroyAllWindows()
