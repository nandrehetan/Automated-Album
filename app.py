import streamlit as st
import os
import shutil
import mediapipe as mp
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
from PIL import Image


map_of_face_encoding=dict({})


def check_folder_existence(folder_path):
    if os.path.isdir(folder_path):
        st.success(f"The path '{folder_path}' is a directory and exists.")
        return 1
    else:
        st.warning(f"The path '{folder_path}' either does not exist or is not a directory.")
        st.warning("Enter full and valid path...")
        return 0

def create_directory(directory_name):
    try:
        shutil.rmtree(directory_name)
        os.mkdir(directory_name)
        st.success(f"Directory '{directory_name}' created successfully.")
    except OSError as e:
        os.mkdir(directory_name)
        st.success(f"Directory '{directory_name}' created successfully.")

def main():
    st.title("EDI PROJECT")

    source_path = st.text_input("Enter folder path:")
    Submit = st.button("Submit")
    source_path =source_path.replace("\\", "/")

    if Submit:
        # Image seperator
        from seperator import seperate
        seperate()

        # For context based images
        new_source_path=source_path+"/images"
        if check_folder_existence(new_source_path):
            from processing import find_all_unique_images,compare_images_2,find_face_encodings,rename
            create_directory("unique_faces")
            unique_path = source_path+"/unique_faces"
            print(new_source_path)
            print(unique_path)
            find_all_unique_images(new_source_path, unique_path)
            print("Unique images completed")
            from clustering import generate
            generate(3)
            print("clusters generated")
            from getscore import find_score,get_score,find_blur_score
            cluster_head_image_score=find_score()
            cluster_blur_image_score=find_blur_score()
            final_score=[]
            for i in range(len(cluster_blur_image_score)):
                curr_=[]
                for j in range (len(cluster_blur_image_score[i])):
                    curr_.append(cluster_blur_image_score[i][j]/cluster_head_image_score[i][j])
                final_score.append(curr_)
            for filename in os.listdir('results'):
                os.remove(f"results/{filename}")
            for i in range(1,len(final_score)+1):
                id=-1
                mx=-1
                for j in range(len(final_score[i-1])):
                    if (final_score[i-1][j]>mx):
                        id=j
                        mx=final_score[i-1][j]
                curr_img=os.listdir(f"clusters/cluster{i}")[id]
                curr_img_path=f"clusters/cluster{i}/{curr_img}"
                shutil.copyfile(curr_img_path, f"results/{curr_img}")

            
        # For context free images
        new_source_path=source_path+"/Natural_images"
        from natural_images_clustering import find_natural_images_score
        if check_folder_existence(new_source_path):
            find_natural_images_score(2,source_path)
        
        print("Done")
        for curr_im in os.listdir('results/'):
            image = Image.open(f"results/{curr_im}")
            st.image(image, caption='Sunrise by the mountains')

           





        


            
            



            



if __name__ == "__main__":
    main()
