#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    	: 08 / 19 / 2022
@Author  	: Zhi Qiao
@Contact	: z.qiao1989@gmail.com
@File    	: pattern_find.py
@Software	: AbsolutePhase
@Desc		: code to find the area of the speckle pattern in the calibrated data
'''

from PIL import Image as tif_image
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import scipy.io as sio

from aps.common.measurment.beamline.wf.func_light import prColor

def normalize(v):
    return (v-np.amin(v)) / (np.amax(v) - np.amin(v))

def image_transfer(img, h_flip=0, v_flip=0, transpose=0, direc='forward'):
    img_new = img.copy()
    if direc == 'forward':
        if h_flip:
            img_new = np.fliplr(img_new)
        if v_flip:
            img_new = np.flipud(img_new)
        if transpose:
            img_new = img_new.transpose()
    elif direc == 'backward':
        if transpose:
            img_new = img_new.transpose()
        if v_flip:
            img_new = np.flipud(img_new)
        if h_flip:
            img_new = np.fliplr(img_new)
        
        
    return img_new


def pattern_match(pattern_path, img_path, pattern_size, img_trans, show_fig=True):
    '''
        use opencv to find matched pattern with the image
        input:
            pattern_path:           path to pattern file (npy)
            img_path:               path to image file (tif)
            pattern_size:             pattern size
            img_trans:              pattern transformation to match image
            show_fig:               show the plot figure or not

    '''
    file_folder = os.path.dirname(img_path)

    p_x = 0.65e-6
    # for 5um pattern
    pattern_pixel = pattern_size
    scale = pattern_pixel / p_x 

    # region to use as template
    row_start = 800
    row_end = 1200
    col_start = 800
    col_end = 1200

    H_flip, V_flip, Dia_transpose = img_trans

    # load data
    prColor('MESSAGE: sample image,  ' + img_path, 'green')
    I_img = tif_image.open(img_path)

    '''
    rotate the image to match the design pattern orientation
    '''
    I_img_rot = image_transfer(np.array(I_img), H_flip, V_flip, Dia_transpose)
    I_img = I_img_rot.astype(np.float32)

    prColor('MESSAGE: pattern image,  ' + pattern_path, 'green')
    I_pattern = np.load(pattern_path)

    I_pattern = normalize(1-I_pattern) * 255
    I_pattern = I_pattern.astype(np.float32)

    I_img_norm = normalize(I_img[row_start:row_end, col_start:col_end]) * 255
    I_img_norm = I_img_norm.astype(np.float32)
    
    h, w = I_pattern.shape

    # All the 6 methods for comparison in a list
    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    meth = 'cv2.TM_CCOEFF'

    temp = np.copy(I_pattern)

    I_pattern_reshape = cv2.resize(temp, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST )
    
    template = I_img_norm
    
    n_template_row, n_template_col = template.shape

    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(I_pattern_reshape,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + n_template_col, top_left[1] + n_template_row)
    color = (255, 0, 0)
    print('region position left top: {}, bottom right: {}'.format(top_left, bottom_right))
    img_small = I_pattern_reshape[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cv2.rectangle(I_pattern_reshape,top_left, bottom_right, color, 1)

    if show_fig:
        plt.figure()
        plt.subplot(131)
        plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result')
        plt.subplot(132)
        plt.imshow(I_pattern_reshape,cmap = 'gray')
        plt.title('full image')
        plt.subplot(133)
        plt.imshow(img_small,cmap = 'gray')
        plt.title('Detected Point')
        plt.suptitle(meth)

        plt.figure()
        plt.subplot(131)
        plt.imshow(img_small,cmap = 'gray')
        plt.title('Matching Result')
        plt.subplot(132)
        plt.imshow(template,cmap = 'gray')
        plt.title('full image')
        plt.subplot(133)
        plt.imshow(255- img_small + I_img_norm)
        plt.title('diff image')

        plt.show()

    '''
        save results
    '''
    n_row, n_col = I_img.shape
    # pad the pattern data to have enough size
    n_row_pattern, n_col_pattern = I_pattern_reshape.shape

    n_pad = np.amax([0, -(top_left[1]-row_start), -(top_left[0]-col_start),
                    bottom_right[1]+n_row-row_end-n_row_pattern,
                    bottom_right[0]+n_col-col_end-n_col_pattern])
    print('padding size: {}'.format(n_pad))

    I_pattern_reshape = np.pad(I_pattern_reshape, (n_pad, n_pad), 'constant', constant_values=(0, 0))

    Img_pattern = I_pattern_reshape[top_left[1]-row_start+n_pad:bottom_right[1]+n_row-row_end+n_pad, 
                                    top_left[0]-col_start+n_pad:bottom_right[0]+n_col-col_end+n_pad]


    Img_pattern = image_transfer(Img_pattern, H_flip, V_flip, Dia_transpose, direc='backward')

    I_img = image_transfer(I_img, H_flip, V_flip, Dia_transpose, direc='backward')

    im = tif_image.fromarray(Img_pattern*I_img)
    im.save(os.path.join(file_folder, 'pattern.tif'))
    prColor('saved pattern image: {}'.format(os.path.join(file_folder, 'pattern.tif')), 'cyan')
    
    sio.savemat(os.path.join(file_folder, 'pattern.mat'), {'I_pattern':Img_pattern, 'I_origin':I_img})
    prColor('saved data: {}'.format(os.path.join(file_folder, 'pattern.mat')), 'cyan')


