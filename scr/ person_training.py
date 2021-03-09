import cv2
import os
import numpy as np

# Asignacion ruta del directorio y su listado
data_path = '../data'
#Eliminacion de .DS_Store para evitar problemas mas adelante
os.remove(data_path +'/.DS_Store')
user_list = os.listdir(data_path)

print('Lista de personas: ', user_list)

#Generacion de listas para almacenar las etiquetas de cada subdirectorio para relacionarlos con cada imagen
labels = []
faces_data = []
label = 0

# Especificacion de ruta del directorio de donde se leeran la imagenes 
for name in user_list:
    person_path = data_path + '/' + name
    print('Leyendo las imágenes')

    for file_name in os.listdir(person_path):
        print('Rostros: ', name + '/' + file_name)
        labels.append(label)
        faces_data.append(cv2.imread(person_path+'/'+file_name,0))
    label += 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Entrenando...")
face_recognizer.train(faces_data, np.array(labels))
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado...")