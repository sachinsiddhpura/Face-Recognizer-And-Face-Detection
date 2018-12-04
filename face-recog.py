import os
import cv2
import numpy as np 
from PIL import Image
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_file=os.path.join(BASE_DIR,'images')

x_roi=[]
y_labels=[]
current_id=0
label_ids={}

for root, dirs, files in os.walk(image_file):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path=os.path.join(root,file)
            label=os.path.basename(root).replace(" ", "-").lower()
            if not label in label_ids:
                label_ids[label]=current_id
                current_id +=1
            id=label_ids[label]

            pil_image=Image.open(path).convert("L")
            size=(550,550)
            fimage=pil_image.resize((550,550),Image.ANTIALIAS)
            image_arry=np.array(pil_image,'uint8')
            faces = face_cascade.detectMultiScale(image_arry,scaleFactor=1.5,minNeighbors=5)

            for (x,y,h,w) in faces:
                roi=image_arry[y:y+h,x:x+w]
                x_roi.append(roi)
                y_labels.append(id)

with open('labels.pickle','wb') as f:
    pickle.dump(label_ids,f)

recognizer.train(x_roi, np.array(y_labels))
recognizer.save('recognizer/trainner.yml')
