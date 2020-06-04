from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np
import cv2
from datetime import datetime
#import winsound
import os
from playsound import playsound
from threading import Thread
import time

Savepath=os.getcwd()
f=open(Savepath+'\\videonumber.txt')
textV=f.readline()
cap = cv2.VideoCapture(int(textV))
f.close()
# files = sys.argv[1:]

net = load_model('model-resnet50-final.h5')
cls_list = ['false', 'other', 'true'] #新增其他
count=30
theTime=datetime.now().strftime("%Y-%m-%d %H-%M-%S")

#path0='C:/Users/sabri/Desktop/Mask-recognize/false'
#path1='C:/Users/sabri/Desktop/Mask-recognize/other'
#path2='C:/Users/sabri/Desktop/Mask-recognize/true' #新增其他
#duration=300
#freq=400

PlaySoundFlag = 0

while(True):
    ret, frame = cap.read()
    # cv2.imshow('frame', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    def play_music():
        #playsound('C:/Users/sabri/Desktop/Mask-recognize/mask.mp3')
        playsound(os.path.join(Savepath,'mask.mp3'))
        #time.sleep(5)
        #return 1
    music_thread=Thread(target=play_music)
    # count=count+1
    #if count%30 == 0:
    f=theTime+'.jpg'
    cv2.imwrite(f, frame)
        #winsound.Beep(freq, duration)
    #else:
    #    continue
    
    img = image.load_img(f, target_size=(224, 224))

    if img is None:
        continue        
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    pred = net.predict(x)[0]
    if pred[2] > 0.95:
        theTime2=datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        f2=theTime2+'.jpg'
        cv2.imwrite(os.path.join(Savepath,'true',f2), frame)
    if pred[1] > 0.95:
        theTime2=datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        f2=theTime2+'.jpg'
        cv2.imwrite(os.path.join(Savepath,'other',f2), frame)
    if pred[0] > 0.95:
        theTime2=datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        f2=theTime2+'.jpg'
        cv2.imwrite(os.path.join(Savepath,'false',f2), frame)
        if PlaySoundFlag == 1:
        #   if music_thread.is_alive():
            count+=1
            if(count==5):
                #print("hello")
                PlaySoundFlag = 0
        else:
            music_thread.start()
            PlaySoundFlag = 1
            count=0
        #winsound.Beep(freq, duration)
    #img = image.load_img(f, target_size=(224, 224))
    
    #if img is None:
    #    continue
    
    top_inds = pred.argsort()[::-1][:5]
    for i in top_inds:
            #print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
            text='{:.3f}  {}'.format(pred[i], cls_list[i])
            cv2.putText(frame, text, (10, 55*(i+1)), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.imshow('frame', frame)
    #   count=count+1
    #   if count%30 == 0:
    #     cv2.imwrite(str(count)+'.jpg', frame)

cap.release()
cv2.destroyAllWindows()
