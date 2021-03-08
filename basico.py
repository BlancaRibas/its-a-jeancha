
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Captura de frames 
    ret, frame = cap.read()

    # Operaciones sobre frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Visualizacion de frames
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)
    
    # Criterio de parada 
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
print('ciao gauper')