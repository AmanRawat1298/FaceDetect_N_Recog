#IMPORTING MODULES
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


#INTIALIZE CAMERA
cap = cv2.VideoCapture(0)

#LAODING THE CLASSIFIER
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


#DATA SET PATH 
dataset_path = "./dataset_face/"




#LOADING ALREADY SAVED DATA
data = np.load(dataset_path +"face_data.npy")
X = data[:, 1:].astype(int)
y = data[:, 0]


#LOADING KNN CLASSIFIER & FITTING IT TO THE EXISTING FACE DATA
model = KNeighborsClassifier(5)
model.fit(X, y)

print("PRESS Q TO EXIT")


while True:
    ret, frame = cap.read()

    if ret:



        #GRAYSCALE IMAGE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        
        #DETECT FACES
        faces = classifier.detectMultiScale(gray)
        areas = []
        
        
        
        for face in faces:
            x, y, w, h = face
            areas.append((w*h, face))

        
        
        if len(faces) > 0:
        
        
            #FACE WITH LARGEST AREAS
            face = max(areas)[1]
            x, y, w, h = face

            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))
            face_flatten = face_img.flatten()



            #LARGEST FACE DATA USED FOR CLASSIFICATION
            res = model.predict([face_flatten])




            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 4)



            #DISPLAYS NAME ON THE RECTANGLE AROUND THE FACE
            cv2.putText(frame, str(res[0]), (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))


        cv2.imshow("video", frame)



    #PRESS Q TO EXIT
    key = cv2.waitKey(1)
    if key & 0xff == ord('q'):
        break