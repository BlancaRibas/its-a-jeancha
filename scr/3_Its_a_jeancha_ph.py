import cv2
import numpy as np
import os

data_path = '../data'

#os.remove(data_path +'/.DS_Store')
image_paths = os.listdir(data_path)
print('image Paths=',image_paths)

#Clasificador de imagenes 
face_classif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modelo/modeloLBPHFace.xml')

#lectura de imagen 
image = cv2.imread('../image.png')

#transformacion a escala de grises
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#Creamos copia de la imagen de entrada
image_aux = gray.copy()

#Aplicamos el clasificador de sobre la imagen 
faces = face_classif.detectMultiScale(gray,
  #Piramide de imagenes
  scaleFactor=1.1,
  #Numero minimo de cuadrados delimitadores vecinos para reconocer rostros
  minNeighbors=5,
  #Tamaño minimo de objetos
  minSize=(30,30),
  #Tamaño maximo del objeto
  maxSize=(200,200))

for (x,y,w,h) in faces:
    rostro = image_aux[y:y+h,x:x+w]
    rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
    result = face_recognizer.predict(rostro)
    cv2.putText(image,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

    if result[1] < 75:
        cv2.putText(image,'{}'.format(image_paths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    else:
        cv2.putText(image,'Not a jeancha!',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
        cv2.rectangle(image, (x,y),(x+w,y+h),(0,0,255),2)

#Visualizar la imagen
cv2.imshow('rectangulos de deteccion',image)
#tiempo de visualizacion con 0 se mantiene hasta presionar una tecla
cv2.waitKey(0)
#cierra las ventanas generadas durante el proceso 
cv2.destroyAllWindows()      
