import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import face_recognition
from PIL import Image
import math 
# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import shutil

map_of_face_encoding=dict({})
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
no_of_unique_faces=len(os.listdir('unique_faces/'))
def find_face_encodings(image):
    # reading image
#     image = cv2.imread(image_path)
#     print(type(image))
    # get face encodings from the image
    face_enc = face_recognition.face_encodings(image)
    # return face encodings
    return face_enc[0]

def compare_images_2(im1,im2):
    image1=0
    image2=0
    curr=cv2.imread(f"unique_faces/{im2}")
    image_1 = find_face_encodings(im1)
    if im2 in map_of_face_encoding:
        image_2=map_of_face_encoding[im2]
    else:
        image_2  = find_face_encodings(curr)
        map_of_face_encoding[im2]=image_2
    try:
        is_same = face_recognition.compare_faces([image_1], image_2)[0]
        return is_same
    except:
        return 0
    
def generate(cluster_number): 
    image_vectors=[]
    for filename in os.listdir('images/'):
        curr_vector=[0]*no_of_unique_faces
        imagePath = f"images/{filename}"
        img = cv2.imread(imagePath)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=9, minSize=(40, 40))
        faces=[]
        for (x, y, w, h) in face:
            im = Image.open(imagePath)
            im1 = im.crop((x, y, x+w, y+h))
            faces.append(im1)
        idx=0
        for face_file in os.listdir('unique_faces/'):
            unique_face_img=cv2.imread(f"unique_faces/{face_file}")
            for face_check in faces:
                curr_im=np.asarray(face_check)
                try:
                    bl=compare_images_2(curr_im,face_file)
                    if (bl):
                        curr_vector[idx]=1
                            
                except:
                    continue
            idx=idx+1
        print(curr_vector)
        image_vectors.append(curr_vector)
            

    kmeans = KMeans(n_clusters=cluster_number, random_state=22)
    kmeans.fit(image_vectors)
    v=kmeans.labels_
    for file in os.listdir('clusters/'):
        for file2 in os.listdir(f"clusters/{file}"):
            os.remove(f"clusters/{file}/{file2}")
        os.rmdir(f"clusters/{file}")
    mx=max(v)
    for i in range(1,mx+2):
        os.mkdir(f"clusters/cluster{i}")
    for i in range(len(v)):
        shutil.copyfile(f"images/img{i+1}.jpg", f"clusters/cluster{v[i]+1}/img{i+1}.jpg")

# print(no_of_unique_faces)     
# generate(3) 
      
