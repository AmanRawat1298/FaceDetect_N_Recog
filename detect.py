#import libraries
import cv2
import numpy as np
import os
from os import path







#INITIALIZE CAMERA
cap = cv2.VideoCapture(0)



#LOAD THE CLASSIFIER
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")



#DEFINING VARIABLES
name = input("Enter the name of the person:")
dataset_path = "./dataset_face/"
counter = 100
face_list = []




while True:
    ret, frame = cap.read()

    if ret:
        #CONVERTING TO GRAYSCALE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #DETECTING FACES IN THE FRAME
        faces = classifier.detectMultiScale(gray)

        areas = []
        for face in faces:
            x, y, w, h = face
            areas.append((w*h, face))
            #print(type(face), face.shape)
        



        #LARGEST FACE'S DATA IS SAVED UNTIL COUNTER IS 0
        if len(faces) > 0:
            face = max(areas)[1]
            x, y, w, h = face

            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))

            face_flatten = face_img.flatten()
            face_list.append(face_flatten)
            counter -= 1
            print("loaded with",counter)
            if counter <= 0:
                break

            cv2.imshow("video", face_img)



    #TO STOP PRESS Q 
    key = cv2.waitKey(1)
    if key & 0xff == ord('q'):
        break



#DATA PREPARATION FOR KNN CLASSIFIER
X = np.array(face_list)
y = np.full((len(X), 1), name)
data = np.hstack([y, X])



#DESTROYING WINDOWS
cap.release()
cv2.destroyAllWindows()



#SAVING THE DATA 
if path.exists(dataset_path):
    
    if path.exists(dataset_path + "face_data.npy"):
        face_data = np.load(dataset_path + "face_data.npy")
        face_data = np.vstack([face_data, data])
        np.save(dataset_path +"face_data.npy", face_data)
        
    else:
        np.save(dataset_path + "face_data.npy", data)
else:
    os.makedirs(dataset_path)
    np.save(dataset_path + "face_data.npy", data)


print(name +"'s Face data is succesfully saved at "+dataset_path+" face_data.npy")

