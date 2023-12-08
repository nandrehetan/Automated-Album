# for loading/processing the images  
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import cv2
import shutil

path = r"Natural_images/"
# this list holds all the image filename
flowers = []

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
  # loops through each file in the directory
    for file in files:
        if file.name.endswith('.jpg'):
          # adds only the image files to the flowers list
            flowers.append(file.name)

model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)


def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

def find_natural_images_score(cluster_count,source_path):
        data = {}
        os.chdir('Natural_images/')
        # lop through each image in the dataset
        for flower in flowers:
            # try to extract the features and update the dictionary
            try:
                feat = extract_features(flower,model)
                data[flower] = feat
            # if something fails, save the extracted features as a pickle file (optional)
            except:
                with open(p,'wb') as file:
                    pickle.dump(data,file)
        # get a list of the filenames
        filenames = np.array(list(data.keys()))

        # get a list of just the features
        feat = np.array(list(data.values()))

        # reshape so that there are 210 samples of 4096 vectors
        feat = feat.reshape(-1,4096)


        pca = PCA(n_components=1, random_state=22)
        pca.fit(feat)
        x = pca.transform(feat)

        kmeans = KMeans(n_clusters=cluster_count, random_state=22)
        kmeans.fit(x)

        # holds the cluster id and the images { id: [images] }
        groups = {}
        for file, cluster in zip(filenames,kmeans.labels_):
            if cluster not in groups.keys():
                groups[cluster] = []
                groups[cluster].append(file)
            else:
                groups[cluster].append(file)
        def variance_of_laplacian(image):
            return cv2.Laplacian(image, cv2.CV_64F).var()

        # Defining the image directory and threshold
        scores_blurry=[]
        for curr_cluster in groups:
            scores_cluster_blurry=[]
            image_directory = ""
            threshold = 100.0

            all_files = groups[curr_cluster]
        #     print(all_files)


            for image_file in all_files:

                imagePath = f"{image_file}"
        #         print(imagePath)
                image = cv2.imread(imagePath)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                fm = variance_of_laplacian(gray)
                scores_cluster_blurry.append(fm)
            scores_blurry.append(scores_cluster_blurry)
                
            #     text = "Not Blurry"

            #     if fm < threshold:
            #         text = "Blurry"

            #     cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            #     cv2.imshow(image)
            #     key = cv2.waitKey(0)

            # cv2.destroyAllWindows()
        os.chdir(source_path)
        for i in range(len(groups)):
            mx=-1
            id=-1
            for j in range(len(scores_blurry[i])):
                if (scores_blurry[i][j]>mx):
                    id=j
                    mx=scores_blurry[i][j]
            curr_img=f"Natural_images/{groups[i][id]}"
            shutil.copyfile(curr_img, f"results/{groups[i][id]}")
            

