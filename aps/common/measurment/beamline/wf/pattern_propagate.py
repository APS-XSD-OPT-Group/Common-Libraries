#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    	: 08 / 19 / 2022
@Author  	: Zhi Qiao
@Contact	: z.qiao1989@gmail.com
@File    	: pattern_propagate.py
@Software	: AbsolutePhase
@Desc		: code to propagate the matched pattern
'''


from PIL import Image as tif_image
import os
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
import scipy.constants as sc
from scipy.ndimage import gaussian_filter

from aps.common.measurment.beamline.wf.func_light import prColor
from aps.common.measurment.beamline.wf.diffraction_process import diffraction_prop


def load_pattern(file_path, print_key=True):
    if os.path.exists(file_path):
        data_content = sio.loadmat(file_path)
        prColor('variable name: {}'.format(data_content.keys()), 'light_purple')
    else:
        prColor('Error: wrong file path!', 'cyan')
        os.exit()

    I_pattern = data_content['I_pattern']
    I_img = data_content['I_origin']
    return I_pattern, I_img

def load_tiff(file_path):
    if os.path.exists(file_path):
        im = tif_image.open(file_path)
        im = np.array(im)
    else:
        prColor('Error: wrong file path!', 'cyan')
        os.exit()

    return im

def normalize(v):
    return (v-np.amin(v)) / (np.amax(v) - np.amin(v))

def PSF_detector(d_reso, p_x, I):
    '''
        the resolution degrades due to PSF
        d_reso:
            detector resolution
        p_x:
            pixel size
        I:
            the data
    '''
    M, N = I.shape
    y_axis = np.arange(-M//2, M//2) * p_x
    x_axis = np.arange(-N//2, N//2) * p_x
    XX, YY = np.meshgrid(x_axis, y_axis)

    sigma = d_reso / (2 * np.sqrt(np.log(2)))
    PSF = np.exp(-(XX**2 + YY**2)/(sigma)**2)

    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))

    return np.abs(ifft2(fft2(I)*fft2(PSF))) 


def PSF_coherence(sigma_h, sigma_v, p_x, I):
    '''
        the resolution degrades due to temporal or spatial coherence
        sigma_h:
            horizontal convolution kernel size
        sigma_v:
            vertical convolution kernel size
        p_x:
            pixel size
        I:
            the data
    '''
    M, N = I.shape
    y_axis = np.arange(-M//2, M//2) * p_x
    x_axis = np.arange(-N//2, N//2) * p_x
    XX, YY = np.meshgrid(x_axis, y_axis)


    PSF = np.exp(-(XX**2/sigma_h**2 + YY**2/sigma_v**2))

    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))

    return np.abs(ifft2(fft2(I)*fft2(PSF)))

def sphrical_wavefront(n_size, distance, p_x, wavelength):
    '''
        n_size:
            the data shape
        distance:
            source distance to speckle
            [v, h]
        p_x:
            pixel size
        wavelength:
            wavelength
    '''
    # generate the simulated incident wavefront from the source
    k = 2 * np.pi / wavelength
    M, N = n_size
    y_axis = np.arange(-M//2, M//2) * p_x
    x_axis = np.arange(-N//2, N//2) * p_x
    XX, YY = np.meshgrid(x_axis, y_axis)

    phase_illum = np.exp(1j*k*(XX**2/(2*distance[1])+YY**2/(2*distance[0])))

    return phase_illum

def pattern_prop(pattern_path, pattern_para=[0.613, 0.5*np.pi], prop_para=[14e3, 200e-3], 
                source_para=[35, 29], show_fig=True):
    '''
        propagate the matched pattern to get pattern-ref image
        input:
                pattern_path:           path to matched pattern file, mat
 
                pattern_para:           [T_transmission, phase]
                                        [transmission of pattern, phase shifting of pattern]
                prop_para:              parameters for propogation
                                        [energy(eV), distance]
                source_para:            source distance
                                        [vertical, horizontal]
    '''
    T_transmission, R_phase = pattern_para

    # energy
    # energy = 14e3
    energy = prop_para[0]
    c_w = sc.value('inverse meter-electron volt relationship') / energy

    # pixel size
    p_x = 0.65e-6
    # distance
    d_prop = prop_para[1]
    # theoretical resolution of the detector
    det_resolution = np.floor(2.2e-6/p_x + 1) * p_x

    # coherence of the source
    source_h = 86.9e-9
    source_v = 39.7e-9

    d_source2pattern_v, d_source2pattern_h = source_para

    d_source2pattern = [d_source2pattern_v, d_source2pattern_h]

    # calculate sigma kernel size for coherence
    sigma_h = source_h/d_source2pattern_h*d_prop
    sigma_v = source_v/d_source2pattern_v*d_prop

    # I_pattern:  pattern matched,   I_img: ref image used to find the matched pattern
    I_pattern, I_ref = load_pattern(pattern_path, print_key=True)
    I_pattern = normalize(I_pattern)

    # propagation
    A_pattern = np.sqrt((1-T_transmission)*I_pattern+T_transmission) * np.exp(1j*I_pattern*R_phase) \
                * sphrical_wavefront(I_pattern.shape, d_source2pattern, p_x, wavelength=c_w)

    N_pad = 1000
    if N_pad == 0:
        A_prop, L_out = diffraction_prop(A_pattern, p_x, d_prop, c_w, 'TF')
    else:
        A_pattern_pad = np.pad(A_pattern, (N_pad, N_pad), mode='constant', constant_values=(0,0))
        A_prop, L_out = diffraction_prop(A_pattern_pad, p_x, d_prop, c_w, 'TF')
        A_prop = A_prop[N_pad:-N_pad, N_pad:-N_pad]
    
    I_prop = np.abs(A_prop) ** 2

    # the detector PSF influence
    I_det = PSF_detector(det_resolution, p_x, I_prop)

    I_coh = PSF_coherence(sigma_h, sigma_v, p_x, I_det)

    # use gaussian fiter for the image
    I_blur = gaussian_filter(I_coh, sigma=3)

    if show_fig:
        plt.figure()
        plt.subplot(221)
        plt.imshow(I_ref[0:200,0:200])
        plt.subplot(222)
        plt.imshow(I_det[0:200,0:200])
        plt.subplot(223)
        plt.imshow(I_coh[0:200,0:200])
        plt.subplot(224)
        plt.imshow(I_blur[0:200,0:200])
        plt.show()

        plt.figure(figsize=(15,8))
        plt.subplot(241)
        plt.title('top right')
        plt.imshow(I_ref[0:200,0:200])
        plt.colorbar()
        plt.subplot(242)
        plt.title('top left')
        plt.imshow(I_ref[0:200,-200:])
        plt.colorbar()
        plt.subplot(243)
        plt.title('bottom right')
        plt.imshow(I_ref[-200:,0:200])
        plt.colorbar()
        plt.subplot(244)
        plt.title('bottom left')
        plt.imshow(I_ref[-200:,-200:])
        plt.colorbar()

        plt.subplot(245)
        plt.imshow(I_coh[0:200,0:200])
        plt.title('top right')
        plt.colorbar()
        plt.subplot(246)
        plt.title('top left')
        plt.imshow(I_coh[0:200,-200:])
        plt.colorbar()
        plt.subplot(247)
        plt.title('bottom right')
        plt.imshow(I_coh[-200:,0:200])
        plt.colorbar()
        plt.subplot(248)
        plt.title('bottom left')
        plt.imshow(I_coh[-200:,-200:])
        plt.colorbar()
        plt.show()

    return I_blur, I_coh, I_det, I_pattern


