### txt转csv文件
import numpy as np
import os
from PIL import Image
import pandas as pd
#from torch import float, int
file_name_path=r"G:\DOTA\train\new_data\train\\"
images_path=file_name_path+"images\\"
labelS_path=file_name_path+"labels\\"
file_name=os.listdir(labelS_path)
dic_name={0: 'large-vehicle',1: 'plane',2: 'ship',3: 'harbor',4: 'tennis-court',5: 'baseball-diamond',6: 'basketball-court',7: 'helicopter'}
# file=open(r"F:\taididata\lastdata\lastdata\manual_label_images\labels\00029.txt","r",encoding='UTF-8')
# file.read()
data=[]
for i in file_name:
    ii=i.split(".")
    # print(i,i[0:6]) 
    cv=Image.open(images_path+ii[0]+".jpg")
    file=open(labelS_path+i,"r",)
    categroy_list=file.read().splitlines(True)
    # print(categroy_list)
    for _ in categroy_list:
        csv_=[]
        list_=[float(i) for i in _.strip().split(" ")]
        # print(list_)
        csv_.append(ii[0]+".jpg")
        csv_.append(int(list_[0]))
        csv_.append(dic_name[int(list_[0])])
        #csv_.append(int(list_[1]*cv.width))
        #csv_.append(int(list_[2]*cv.height))
        csv_.append(int((list_[1]*cv.width)-(list_[3]*cv.width/2)))
        csv_.append(int((list_[2]*cv.height)-(list_[4]*cv.height/2)))
        csv_.append(int((list_[1]*cv.width)+(list_[3]*cv.width/2)))
        csv_.append(int((list_[2]*cv.height)+(list_[4]*cv.height/2)))
        
        # csv_
        data.append(csv_)
data_csv=pd.DataFrame(data,columns=['文件名','分类号','类型','左上角x坐标','左上角y坐标','右下角x坐标','右下角y坐标'])
data_csv.head(3)
data_csv.to_csv("G://DOTA//new_data.csv",encoding='utf-8')