import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time
import os





def get_score(filename):
    scores_of_image=[]
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    
    for curr_im in os.listdir(f"clusters/{filename}"):
        # Load face detection model
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

        scores_of_face=[]
#        print(curr_im)
        imagepath = f"clusters/{filename}/{curr_im}"
        image = cv2.imread(imagepath)
        image2=cv2.imread(imagepath)

        # Convert image to RGB (required for face mesh)
        image_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        # Detect faces using Haar Cascade
        faces = face_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=9, minSize=(40, 40))
#         print(len(faces))

        # Loop over each detected face
        cnt=0

        for (x1, y1, w1, h1) in faces:
            # Load face mesh model
#             print(x1,y1,w1,h1)
            # Crop face using PIL
            im = Image.open(imagepath).convert('L')
            im_face = im.crop((x1, y1, x1+w1, y1+h1))
            # im_face=cv2.resize(im_face,(300,300))
#             print("here")

            # Convert PIL image back to OpenCV format
            im_face_cv2 = np.array(im_face)
            im_face_cv2 = cv2.resize(im_face_cv2,(1000,1000))
            im_face_cv2 = cv2.cvtColor(im_face_cv2, cv2.COLOR_GRAY2BGR)

            # Face Mesh
            start = time.time()
            results = face_mesh.process(image_rgb)
#             print("here")

            if results.multi_face_landmarks:
#                 print("here")
                for face_landmarks in results.multi_face_landmarks:
                    # ... (your existing head pose estimation code)

                    img_h, img_w, img_c = image.shape
                    face_3d = []
                    face_2d = []

                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append([x, y])

                            # Get the 3D Coordinates
                            face_3d.append([x, y, lm.z])       

                    # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)

                    # Convert it to the NumPy array
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # The camera matrix
                    focal_length = 1 * img_w

                    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                            [0, focal_length, img_w / 2],
                                            [0, 0, 1]])

                    # The distortion parameters
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360
#                     print(x,y,z)


                    # See where the user's head tilting
                    if y < -10:
                        text = "Looking Left"
                    elif y > 10:
                        text = "Looking Right"
                    elif x < -10:
                        text = "Looking Down"
                    elif x > 10:
                        text = "Looking Up"
                    else:
                        text = "Forward"

                    # Display nose direction for each face
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                    cv2.line(im_face_cv2, p1, p2, (255, 0, 0), 3)

                    # Add text on the cropped face image
                    cv2.putText(im_face_cv2, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(im_face_cv2, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(im_face_cv2, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(im_face_cv2, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                end = time.time()
                totalTime = end - start

                if totalTime > 0:
                    fps = 1 / totalTime
                    cv2.putText(im_face_cv2, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                curr=x**2+y**2+z**2
                print(x,y,z)
                scores_of_face.append(curr)
        try:
            scores_of_image.append(sum(scores_of_face)/len(scores_of_face))
        except:
            scores_of_image.append(10000000000)
    return scores_of_image
        
            
def find_score():               
    cluster_scores=[]
    for filename in os.listdir('clusters/'):
        l=get_score(filename)
        cluster_scores.append(l)
    return cluster_scores

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def find_blur_score():
    scores_blurry=[]
    for curr_cluster in os.listdir('clusters/'):
        scores_cluster_blurry=[]
        image_directory = f"clusters/{curr_cluster}/"
        threshold = 100.0

        all_files = os.listdir(image_directory)
        image_files = [file for file in all_files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]


        for image_file in image_files:

            imagePath = os.path.join(image_directory, image_file)
            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(gray)
            scores_cluster_blurry.append(fm)
            print(fm)
        scores_blurry.append(scores_cluster_blurry)
    return scores_blurry



        