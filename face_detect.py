import numpy as np 
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./recognizer/trainner.yml')

labels={"person_name":1}
with open('labels.pickle','rb') as f:
    orignal_l=pickle.load(f)
    labels={v:k for k,v in orignal_l.items()}

cap=cv2.VideoCapture(0)

while(True):

    ret, frame =cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_frame=frame[y:y+h,x:x+w]

        id, conf=recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

        img='image.png'
        cv2.imwrite(img,roi_gray)

        color=(255,0,0)
        stroke =2
        end_x=x+w
        end_y=y+h 
        cv2.rectangle(frame,(x,y),(end_x,end_y),color,stroke)
        eye=smile_cascade.detectMultiScale(roi_gray)
        for (ex,ey,eh,ew) in eye:
            cv2.rectangle(roi_frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('this is me',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.release()
cv2.destroyAllWindows()