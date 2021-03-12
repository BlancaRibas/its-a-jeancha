import cv2
import os

data_path = '../data'
#os.remove(data_path +'/.DS_Store')
image_paths = os.listdir(data_path)
print('image Paths=',image_paths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modelo/modeloLBPHFace.xml')

cap = cv2.VideoCapture('../test2.mov')

face_classif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
    ret,frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = gray.copy()

    faces = face_classif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        rostro = aux_frame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
        #.predict predice una etiqueta y la confianza para una imagen 
        result = face_recognizer.predict(rostro)

        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
        
        if result[1] < 75:
            # Persona conocida
            cv2.putText(frame,'{}'.format(image_paths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            # Persona desconocida
            cv2.putText(frame,'Not a jeancha!',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
