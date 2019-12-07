#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This is for data augmentation of color-mask pair images by moriitkys


# --- Tools and parameters ---
import os
import numpy as np
import cv2
import glob
import random
import matplotlib.pyplot as plt

do_reverse = True
do_gamma_correction = True
do_add_noise = True
do_cut_out = True
do_deformation = True
irate = 2 # inflation rate for deformation

os.chdir('../')
pathc = os.getcwd()
pathc = pathc.replace("\\", '/')
#print(pathc)

img_output_folder = '/augmentation/img_out'
mask_output_folder = '/augmentation/mask_out'

path_mrcnn_dataset = "path_mrcnn_dataset"#You should change here

f = open('./augmentation/list.txt', 'w')
def imgSave(img, mask):
    global n
    n=n+1
    filename1 = pathc+img_output_folder+'/img'+str(n).zfill(4)+'.png'
    filename2 = pathc+mask_output_folder+'/img'+str(n).zfill(4)+'_mask.png'
    f.write( path_mrcnn_dataset + "\img" + str(n).zfill(4) + '.png'+'\n' )
    cv2.imwrite(filename1, img)
    cv2.imwrite(filename2, mask)
    
if len(glob.glob(pathc + img_output_folder + '/*')) > 0 or len(glob.glob(pathc + mask_output_folder + '/*')) > 0:
    f.close()
    raise TypeError('Your output directory is not empty')


#You can get color and mask images, and list.txt
img_input_folder = '/augmentation/img_in'
mask_input_folder = '/augmentation/mask_in'

imgs = glob.glob(pathc + img_input_folder + '/*')
masks = glob.glob(pathc + mask_input_folder + '/*')

def horizontalFlip(img):
    img = img[:,::-1,:]
    return img

def luminanceUp(img):
    dst = img*1.3
    return dst
def luminanceDown(img):
    dst = img*0.7
    return dst

def addNoise(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    row,col,ch = img.shape
    if np.random.uniform(0, 1) > 0.7:
        # 白
        pts_x = np.random.randint(0, col-1 , 400) #0から(col-1)までの乱数を千個作る
        pts_y = np.random.randint(0, row-1 , 400)
        if row < 600:
            pts_x = np.random.randint(0, col-1 , 100) #0から(col-1)までの乱数を千個作る
            pts_y = np.random.randint(0, row-1 , 100)
        img[(pts_y,pts_x)] = (255,255,255) #y,xの順番になることに注意

        # 黒
        pts_x = np.random.randint(0, col-1 , 400)
        pts_y = np.random.randint(0, row-1 , 400)
        if row < 600:
            pts_x = np.random.randint(0, col-1 , 100) #0から(col-1)までの乱数を千個作る
            pts_y = np.random.randint(0, row-1 , 100)
        img[(pts_y,pts_x)] = (0,0,0)

        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    else:
        pts_x = np.random.randint(0, col-1 , 25)
        pts_y = np.random.randint(0, row-1 , 25)
        img[(pts_y,pts_x)] = (255,255,255)
        pts_x = np.random.randint(0, col-1 , 25)
        pts_y = np.random.randint(0, row-1 , 25)
        img[(pts_y,pts_x)] = (0,0,0)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    return img

def cutOut(img):
    global mask
    rows,cols,ch = img.shape
    #rows is height, cols is width
    imgco = img
    rn1 = random.randint(0, int(cols))#In python3, these should be int() 
    rn2 = random.randint(0, int(rows))
    w = random.randint(0, int(cols/3))
    h = random.randint(0, int(rows/3))
    cval = random.randint(0, 255)
    cv2.rectangle(imgco, (rn1, rn2), (rn1 + w, rn2 + h), (cval, cval, cval), -1)
    cv2.rectangle(mask, (rn1, rn2), (rn1 + w, rn2 + h), (0, 0, 0), -1)

    return imgco

def homographyTrans(img, mask):
    rows,cols,ch = img.shape
    rn1 = random.randint(0, int(cols/5))
    rn2 = random.randint(0, int(cols/5))
    rn3 = random.randint(int(cols*4/5), cols)
    rn4 = random.randint(int(cols*4/5), cols)
    rn5 = random.randint(0, int(rows/5))
    rn6 = random.randint(0, int(rows/5))
    rn7 = random.randint(int(rows*4/5), rows)
    rn8 = random.randint(int(rows*4/5), rows)
    
    pts1 = np.float32([[rn1,rn5],[rn3,rn6],[rn2,rn7],[rn4,rn8]])
    pts2 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(cols,rows))
    mask_tmp = cv2.warpPerspective(mask,M,(cols,rows),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_TRANSPARENT)

    imgSave(dst, mask_tmp)

n = 0
cnt = 0
# --- Save loaded images ---
for cnt in range(len(imgs)):
    image = cv2.imread(imgs[cnt])
    mask = cv2.imread(masks[cnt])
    #print(imgs[cnt])
    #print(masks[cnt])
    for i in range(len(masks)):
        if imgs[cnt].replace('img_in', 'in') == masks[i].replace('mask_in', 'in') :
            mask = cv2.imread(masks[i])
            break
    imgSave(image, mask)
    cnt = cnt + 1
# --- Save reversed images ---
if do_reverse == True:
    cnt = 0
    img_input_folder = '/augmentation/img_out'
    mask_input_folder = '/augmentation/mask_out'
    imgs = glob.glob(pathc + img_input_folder + '/*')
    masks = glob.glob(pathc + mask_input_folder + '/*')
    for cnt in range(len(imgs)):
        image = cv2.imread(imgs[cnt])
        mask = cv2.imread(masks[cnt])
        #print(imgs[cnt])
        #print(masks[cnt])
        for i in range(len(masks)):
            if imgs[cnt].replace('img_in', 'in') == masks[i].replace('mask_in', 'in') :
                mask = cv2.imread(masks[i])
                break

        image_rev = horizontalFlip(image)
        mask_rev = horizontalFlip(mask)
        imgSave(image_rev, mask_rev)
        cnt = cnt + 1
    
# --- Save gamma corrected images ---
if do_gamma_correction == True:
    cnt = 0
    img_input_folder = '/augmentation/img_out'
    mask_input_folder = '/augmentation/mask_out'
    imgs = glob.glob(pathc + img_input_folder + '/*')
    masks = glob.glob(pathc + mask_input_folder + '/*')
    for cnt in range(len(imgs)):
        image = cv2.imread(imgs[cnt])
        mask = cv2.imread(masks[cnt])
        #print(imgs[cnt])
        #print(masks[cnt])
        for i in range(len(masks)):
            mask_name = masks[i].replace('_mask.png', '.png')
            if imgs[cnt].replace('img_out', 'in') == mask_name.replace('mask_out', 'in') :
                mask = cv2.imread(masks[i])
                break
        image_lumup = luminanceUp(image)
        image_lumdown = luminanceDown(image)

        imgSave(image_lumup, mask)
        imgSave(image_lumdown, mask)
        cnt = cnt + 1
        
# --- Save noised images ---
if do_add_noise == True:
    cnt = 0
    img_input_folder = '/augmentation/img_out'
    mask_input_folder = '/augmentation/mask_out'
    imgs = glob.glob(pathc + img_input_folder + '/*')
    masks = glob.glob(pathc + mask_input_folder + '/*')
    for cnt in range(len(imgs)):
        image = cv2.imread(imgs[cnt])
        mask = cv2.imread(masks[cnt])
        #print(imgs[cnt])
        #print(masks[cnt])
        for i in range(len(masks)):
            mask_name = masks[i].replace('_mask.png', '.png')
            if imgs[cnt].replace('img_out', 'in') == mask_name.replace('mask_out', 'in') :
                mask = cv2.imread(masks[i])
                break
        image_pn = addNoise(image)
        imgSave(image_pn, mask)

# --- Save cut-out images ---
if do_cut_out == True:
    cnt = 0
    img_input_folder = '/augmentation/img_out'
    mask_input_folder = '/augmentation/mask_out'
    imgs = glob.glob(pathc + img_input_folder + '/*')
    masks = glob.glob(pathc + mask_input_folder + '/*')
    for cnt in range(len(imgs)):
        image = cv2.imread(imgs[cnt])
        mask = cv2.imread(masks[cnt])
        #print(imgs[cnt])
        #print(masks[cnt])
        for i in range(len(masks)):
            mask_name = masks[i].replace('_mask.png', '.png')
            if imgs[cnt].replace('img_out', 'in') == mask_name.replace('mask_out', 'in') :
                mask = cv2.imread(masks[i])
                break
        image_co = cutOut(image)
        imgSave(image_co, mask)

# --- Save deformed images ---
if do_deformation == True:
    cnt = 0
    img_input_folder = '/augmentation/img_out'
    mask_input_folder = '/augmentation/mask_out'
    imgs = glob.glob(pathc + img_input_folder + '/*')
    masks = glob.glob(pathc + mask_input_folder + '/*')
    for cnt in range(len(imgs)):
        image = cv2.imread(imgs[cnt])
        mask = cv2.imread(masks[cnt])
        #print(imgs[cnt])
        #print(masks[cnt])
        for i in range(len(masks)):
            mask_name = masks[i].replace('_mask.png', '.png')
            if imgs[cnt].replace('img_out', 'in') == mask_name.replace('mask_out', 'in') :
                mask = cv2.imread(masks[i])
                break

        for i in range(irate):
            homographyTrans(image, mask)
f.close()
