from cProfile import label
from PIL import Image
import os
import numpy as np
import cv2
import pickle




face_cascade = cv2.CascadeClassifier('full location of cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()



BASE_DIR =os.path.dirname(os.path.abspath(__file__))
image_dir= os.path.join(BASE_DIR, "images")
current_id=0
label_ids={}
y_labels=[]
x_train=[] 

for root, dirs ,files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith("jpg"):
            path = os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            # print(label,path)


            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            
            id_= label_ids[label]
            # print(label_ids)
            # y_labels.append(label) #some no.
            # x_train.append(path)  #verify this image and then turn it into NumPy array ,GRAY
            pill_image= Image.open(path).convert("L")#into grayscale
            image_array =np.array(pill_image,"uint8")#into nunpy array
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5 ,minNeighbors=5)
            
            for (x,y,w,h) in faces:
                roi =image_array[y:y+h , x:x+w]
                x_train.append(roi)#arrays
                y_labels.append(id_)#y labels into numpy arrays


 
# print(y_labels)
# print(x_train)


with open("labels.pickle",'wb') as f:#save ids
     pickle.dump(label_ids,f)


recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")
