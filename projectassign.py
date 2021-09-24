import numpy as np 
import cv2
import imutils
import time

prototxt="MobileNetSSD_deploy.prototxt.txt"
model="MobileNetSSD_deploy.caffemodel"
confThrash=0.2

CLASSES=["background","aeroplane","bicycle","bird","boat",
"bottel","bus","car","cat","chair","cow","diningtable",
"dog","horse","motorbikes","person","pottedplant","shee",
"sofa","train","tvmonitor"]

COLORS=np.random.uniform(0,255,size=(len(CLASSES),3))

print("loading model.......")

net=cv2.dnn.readNetFromCaffe(prototxt,model)

print("Model loaded")

print("starting camera feed...")

vc=cv2.VideoCapture(0)
time.sleep(2.0)

while True:

    _,frame=vc.read()
    frame=imutils.resize(frame,width=500)

    (h,w)=frame.shape[:2]

    imResizeBlob=cv2.resize(frame,(300,300))
    blob=cv2.cv2.dnn.blobFromImage(imResizeBlob ,0.007843,(300,300),127.5)

    net.setInput(blob)

    detection=net.forward()

    detshape=detection.shape[2]

    for i in np.arange(0,detshape):

        confidence=detection[0,0,i,2]
        ##for getting values
        print(detection[0,0,i,2])
        if confidence > confThrash:
            idx=int(detection[0,0,i,1])
            box=detection[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")

            label="{}:{:2f}%".format(CLASSES[idx],confidence*100)
            cv2.rectangle(frame,(startX,startY),(endX,endY),COLORS[idx],2)

            if startY-15 >15:
                y=startY-15
            else:
                startY+15

            if idx==5.0:
                            cv2.putText(frame,label+"I need Water",(startX,startY),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx])
            else:
                            cv2.putText(frame,label,(startX,startY),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx])
                            
            

    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    if key==27:
        break
vs.release()

cv2.destroyAllWindows()












