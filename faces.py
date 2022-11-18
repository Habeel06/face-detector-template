import numpy as np
import cv2
import pickle



face_cascade = cv2.CascadeClassifier('write full location/cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('write full location of /trainer.yml')

labels={'person_name':1}
with open("C:/Users/HP/Desktop/MAIN/Scripts/PROJECTS/FaceDetector/labels.pickle",'rb') as f:#save ids
    op_labels=pickle.load(f)
    labels={v:k for k,v in op_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret ,frame = cap.read()#capturing fram by frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#grayscale
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5 ,minNeighbors=5)
    for(x,y,w,h) in faces:
        # print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]#region of interest
        roi_color = frame[y:y+h, x:x+w]


        #recognize?
        id_, conf=recognizer.predict(roi_gray)
        if conf>=45 and conf <=85:#conf-confidence
            # print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color =(255,255,255)
            stroke=1
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)



        
        img_item = 'my-image.png'
        cv2.imwrite(img_item,roi_gray)
        
        end_cord_x=x+w #box
        end_cord_y=-y+h
        color = (225,0,0)
        stroke = 2
        cv2.rectangle(frame, (x,y),(end_cord_x,end_cord_y),color ,stroke)
 
    cv2.imshow('frame', frame)#displaying video
    if cv2.waitKey(20) & 0xFF == ord('q'):
     break




cap.release()
cv2.destroyAllWindows()   
   
