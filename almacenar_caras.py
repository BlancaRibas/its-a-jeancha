import cv2
import numpy as np

#Clasificador de imagenes 
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#lectura de imagen 
image = cv2.imread('image.png')
#Creamos copia de la imagen de entrada
image_aux = image.copy()

#transformacion a escala de grises
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)



#Aplicamos el clasificador de sobre la imagen 
faces = faceClassif.detectMultiScale(gray,
  #Piramide de imagenes
  scaleFactor=1.1,
  #Numero minimo de cuadrados delimitadores vecinos para reconocer rostros
  minNeighbors=5,
  #Tamaño minimo de objetos
  minSize=(30,30),
  #Tamaño maximo del objeto
  maxSize=(200,200))


#Contador de rostros, para generar un identificador de cada cara almacenada 
count = 0 

#En caso de detectar un rostro almacenamos en un rectangulo las caras
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    #Edicion de la cara encontrada
    #Recortar la cara de la imagen de entrada
    rostro = image_aux[y:y+h,x:x+w]
    #Redimensionar la cara encontrada
    rostro = cv2.resize(rostro,(150,150),interpolation = cv2.INTER_CUBIC)
    #Alamacenar cara 
    cv2.imwrite('images/rostro_{}.jpg'.format(count),rostro)
    count += 1     
 
    cv2.imshow('rectangulos de deteccion',rostro)

    #Visualizar la imagen
    cv2.imshow('rectangulos de deteccion',image)
    #tiempo de visualizacion con 0 se mantiene hasta presionar una tecla
    #cv2.waitKey(0)
#cierra las ventanas generadas durante el proceso 
cv2.destroyAllWindows()