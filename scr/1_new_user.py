import cv2
import os
import imutils

# Asignamos a personName el nombre la persona donde se almacenara los rostros 
person_name = 'jeancha'

#Especificamos la ruta del directorio data creado a "mano"
data_path = '../data' 
person_path = data_path + '/' + person_name
print(person_name)

#Si no existe un directorio a analizar crearemos uno 
if not os.path.exists(person_path):
    print('Carpeta creada: ',person_path)
    os.makedirs(person_path)

#Video de donde obtendremos la coleccion de rostros 
cap = cv2.VideoCapture('../video/video9.mov')

#Iniciamos el detector de reostros con haarcascades
face_classif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# leemos cada fotograma del video 
count = 0
while True:
    ret, frame = cap.read()
    if ret == False: 
        break
    #Redimensionamos el video
    frame =  imutils.resize(frame, width=650)
    
    #Transformacion a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = frame.copy()
    
    #Aplicamos el clasificador de sobre la imagen 
    faces = face_classif.detectMultiScale(gray,1.3,5)
    
    #En caso de detectar un rostro almacenamos en un rectangulo las caras
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        
        #Edicion de la cara encontrada
        #Recortar la cara de la imagen de entrada
        rostro = aux_frame[y:y+h,x:x+w]
        #redimensionar recorte (150X150pix)
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        #Almacenar cara recortada
        cv2.imwrite(person_path + '/rostro9_{}.jpg'.format(count),rostro)
        count += 1
    
    #Visualizar video
    #cv2.imshow('frame',frame)
    
    #Almacenamiento de 2000rostros 
    k =  cv2.waitKey(1)
    if k == 27 or count >= 300:
        break
        
cap.release()
cv2.destroyAllWindows()
    