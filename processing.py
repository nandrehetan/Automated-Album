import os
import face_recognition
import numpy as np
from PIL import Image
import math
import cv2


map_of_face_encoding=dict({})
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

def rename(DATA_PATH):
    cnt=1
    for filename in os.listdir(DATA_PATH):
        src=DATA_PATH+f"/{filename}"
        dst=DATA_PATH+f"/img{cnt}.jpg"
        cnt=cnt+1
        # print(src)
        try:
            os.rename(src,dst)
        except:
            continue


def find_face_encodings(image):
    # get face encodings from the image
    face_enc = face_recognition.face_encodings(image)
    return face_enc[0]

def compare_images_2(im1,im2):
    try:
        image1=0
        image2=0
        curr=cv2.imread(f"unique_faces/{im2}")
        image_1 = find_face_encodings(im1)
        if im2 in map_of_face_encoding:
            image_2=map_of_face_encoding[im2]
        else:
            image_2  = find_face_encodings(curr)
            map_of_face_encoding[im2]=image_2
        is_same = face_recognition.compare_faces([image_1], image_2)[0]
        return is_same
    except:
        return 1
    

def find_all_unique_images(DATA_PATH, UNIQUE_PATH):

    rename(DATA_PATH)
    cnt = 1

    for file in os.listdir(UNIQUE_PATH):
        os.remove(os.path.join(UNIQUE_PATH, file))
    # print("here ",DATA_PATH)
    for filename in os.listdir(DATA_PATH):
        image_path = DATA_PATH+f"/{filename}"

        # Try to read the image
        print(image_path)
        img = cv2.imread(image_path)

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=9, minSize=(40, 40))
        faces = []
      
        for (x, y, w, h) in face:
            im = Image.open(image_path)
            faces.append(im.crop((x, y, x + w, y + h)))
        print(len(faces))
        for face in faces:
            print(cnt)
            found = 1
            curr_im = np.asarray(face)
            for face_img in os.listdir(UNIQUE_PATH):


                # curr = cv2.imread(os.path.join(UNIQUE_PATH, face_img))
                bl = compare_images_2(curr_im, face_img)
                if bl:
                    found = 0

            if found == 1:
                face.save(os.path.join(UNIQUE_PATH, f"{cnt}.jpg"))
                cnt = cnt + 1
# find_all_unique_images("C:/Users/anike/OneDrive/Desktop/EDI_TY/images","C:/Users/anike/OneDrive/Desktop/EDI_TY/unique_faces")