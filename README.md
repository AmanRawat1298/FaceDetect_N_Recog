# FaceDetect_N_Recog

## Requirments
   1. OPEN CV 
      * pip install opencv-python
   2. OS 
      * pip install os-sys
   3. NUMPY  
      * pip install numpy
   
   

## Getting Started
   * Just clone the repo to your machine. Run detect.py
   * The folder must have haarcascade_frontalface_default.xml file
   * The haarcascade_frontalface_default. xml is a haar cascade designed by OpenCV to detect the frontal face. This haar cascade is available on github . A Haar Cascade works by training the cascade on thousands of negative images with the positive image superimposed on it.

   * It will open your webcam , and capture your face data till counter is 0 (counter initialized with 100 & decrements)
   * If its running for the first time , it will create a dataset_face repo where your face data that is captured         from your webcam is stored
   * Enter your name in terminal
   * After this run recog.py, whenever your face is detected it will show the respective name on top of it!
