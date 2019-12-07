#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This code is for making datasets for mask r-cnn by moriitkys


# --- Tools and Parameters---
import cv2
import numpy as np
import os
import glob
os.chdir('../')
pathc = os.getcwd()
pathc = pathc.replace("\\", '/')
#print(pathc)

def imgSave(img, output_folder):
    global n
    n=n+1
    filename = pathc+'/'+output_folder+'/img'+str(n).zfill(4)+'.png'
    #print(filename1)
    cv2.imwrite(filename, img)

#fillblack
#(255,255,255)以外のピクセルをすべて黒にする

input_folder = 'dw_fb/fb_in'
output_folder = 'dw_fb/fb_out'
masks = glob.glob(pathc + '/' + input_folder + '/*')

n = 0

for f in masks:
    src = cv2.imread(f)
    out = src.copy()

    height, width = src.shape[:2]

    for j in range(0, height):
        for i in range(0, width):
            b_src = src.item(j,i,0)
            g_src = src.item(j,i,1)
            r_src = src.item(j,i,2)
            if b_src == 255 and g_src == 255 and r_src == 255:
                out.itemset((j, i, 0), 255)
                out.itemset((j, i, 1), 255)
                out.itemset((j, i, 2), 255)
            else:
                out.itemset((j, i, 0), 0)
                out.itemset((j, i, 1), 0)
                out.itemset((j, i, 2), 0)
    imgSave(out, output_folder)
