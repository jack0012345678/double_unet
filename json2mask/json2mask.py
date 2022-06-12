import json
import os
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
for filename in os.listdir("./Train_Annotations/"):
    with open("./Train_Annotations/"+filename, "r") as read_file:
        data = json.load(read_file)
    file_name = Path(filename).stem
    
    all_file_names=list(data.keys())#['version', 'flags', 'shapes', 'imagePath', 'imageData', 'imageHeight', 'imageWidth']
  
    
    num=len(data[all_file_names[2]])#檔案中的stas數
    a=0
    for i in all_file_names:
        if (i=='imagePath'):
            image_name=data[all_file_names[a]]
        a+=1
    print(image_name)
    img=np.asarray(PIL.Image.open('Train_Images/'+image_name))
    all_p = []

    for j in range(len(data[all_file_names[2]])):
        shape_x = []
        shape_y = []
        for i in range(len(data[all_file_names[2]][j]['points'])):
            shape_x.append(data[all_file_names[2]][j]['points'][i][0])
            shape_y.append(data[all_file_names[2]][j]['points'][i][1])
        #all_xy.append((shape_x,shape_y))
        ab=np.stack((shape_x, shape_y), axis=1)
        ab = np.int0(ab)
        
        img2=cv2.drawContours(img, [ab], -1, (255,255,255), -1)
        mask = np.zeros((img.shape[0],img.shape[1]))
        cv2.drawContours(mask, [ab], -1, 255, -1)
        #resimg = cv2.add()
        all_p.append(mask)
        
        
        
    res = np.zeros((img.shape[0],img.shape[1]))
    for i in all_p:
        res = cv2.add(res,i)

    cv2.imwrite('masks/'+file_name+'.png',res.astype(np.uint8))  #改存成png


