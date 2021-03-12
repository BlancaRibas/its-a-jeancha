

import pandas as pd
import streamlit as st
from PIL import Image
import os 
import cv2


# Portada
imagen = Image.open("../portadas/portada banner verdes sin persona.jpg")
st.image(imagen)
   
uploaded_file = st.file_uploader("Introduce una foto", type = ['jpeg', 'jpg', 'png'])
        
if uploaded_file:
    #st.write('File successfully uploaded')
    img = Image.open(uploaded_file)
    img.save('analisis/a.jpg')
    st.image(img)
    if st.button('Analizar YA'):

        st.write('.....')
         
    #os.system('open analisis')    
    #cv2.imwrite('analisis1',img)
    imagencita = cv2.imread('analisis/a.jpg')
        
    #transformacion a escala de grises
    gray = cv2.cvtColor(imagencita,cv2.COLOR_BGR2GRAY)
    #Creamos copia de la imagen de entrada
    image_aux = gray.copy()
        
    #Clasificador de imagenes 
        
    face_classif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('modelo/modeloLBPHFace.xml')
        


    #Aplicamos el clasificador de sobre la imagen 
    faces = face_classif.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30,30), maxSize=(200,200))

    for (x,y,w,h) in faces:
        rostro = image_aux[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)
        cv2.putText(imagencita,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
            
        imagen_no = Image.open("../portadas/BOTONrojoprueba.png")
        imagen_si = Image.open("../portadas/BOTONazulprueba.png")
            
            

        if result[1] < 70:
            st.image(imagen_si)
            #resultado = cv2.putText(image,'{}'.format(image_paths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            #imagen = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        else:
            st.image(imagen_no)


