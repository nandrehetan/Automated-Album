import cv2
import PIL
from PIL import Image
import os
import numpy as np

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

import shutil
def seperate():
    for file in os.listdir('Natural_images/'):
            os.remove(f"Natural_images/{file}")
    for file in os.listdir('images/'):
            os.remove(f"images/{file}")
    for filename in os.listdir("dataset/"):
        img = cv2.imread(f"dataset/{filename}")
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=9, minSize=(40, 40))
        faces=[]
        imagePath=f"dataset/{filename}"
        for (x, y, w, h) in face:
            im = Image.open(imagePath)
            im1 = im.crop((x, y, x+w, y+h))
            faces.append(im1)
        if (len(faces)==0):
            shutil.copyfile(f"dataset/{filename}", f"Natural_images/{filename}")   
        else:
            shutil.copyfile(f"dataset/{filename}", f"images/{filename}") 
            