'''
    single-shot absolute measurement based on the simulated reference from mask distribution
    2022/5/11
    by Zhi Qiao
'''

import os
import sys

import numpy as np
import argparse
import scipy.constants as sc
import cv2
import scipy.ndimage as snd
import scipy.signal as ssignal
from matplotlib import pyplot as plt

from aps.common.measurment.beamline.wf.diffraction_process import prop_TF_2d
from aps.common.measurment.beamline.wf.WXST import WXST, save_data, save_figure, save_figure_1D
from aps.common.measurment.beamline.wf.func import prColor, load_image, slop_tracking, write_json, auto_crop, image_align
from aps.common.measurment.beamline.wf.gui_func import crop_gui
from aps.common.measurment.beamline.wf.utils import fft2, ifft2
from aps.common.measurment.beamline.wf.integration import frankotchellappa

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class pattern_search:
    '''
        search the pattern position from the image
        find the relative movement, scales, rotation, and blurring effect for detector and coherence
    '''
    def __init__(self, ini_para=None) -> None:
        if ini_para is None:
            self.p_x = 0.65e-6
            self.pattern_pixel = 4.985e-6
            self.pattern_transmission = 0.613
            self.energy = 20e3
            delta_mask = self.get_delta(self.energy)
            self.c_w = sc.value(
                'inverse meter-electron volt relationship') / self.energy
            self.pattern_phase = 1.5e-6 * delta_mask / self.c_w * 2 * np.pi
            self.d_propagation = 462e-3
            # 28ID
            self.source_distance_v = 40
            self.source_distance_h = 30
            self.source_v = 10e-6
            self.source_h = 277e-6
            self.det_res = 1.5e-6
            self.prop_mode = 'RS'
            # if correct scales or not
            self.correct_scale = False
            self.show_alignFigure = False
            self.det_array = [2160, 2560]
        else:
            self.p_x = ini_para['p_x']
            self.pattern_pixel = ini_para['pattern_size']
            self.pattern_transmission = ini_para['pattern_T']
            self.energy = ini_para['energy']
            delta_mask = self.get_delta(self.energy)
            self.c_w = sc.value(
                'inverse meter-electron volt relationship') / self.energy
            self.pattern_phase = ini_para[
                'pattern_thickness'] * delta_mask / self.c_w * 2 * np.pi
            self.d_propagation = ini_para['d_prop']
            # 28ID
            self.source_distance_v = ini_para['d_sv']
            self.source_distance_h = ini_para['d_sh']
            self.source_v = ini_para['sv']
            self.source_h = ini_para['sh']
            self.det_res = ini_para['det_res']
            self.prop_mode = ini_para['propagator']
            # if correct scales or not
            self.correct_scale = ini_para['correct_scale']
            self.show_alignFigure = ini_para['showAlignFigure']
            self.det_array = ini_para['det_size']

    def get_delta(self, energy):
        file_delta = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Au_delta.npy')
        data = np.load(file_delta)
        x_ev = data[:, 0]
        delta_line = data[:, 1]
        delta = np.interp(energy, x_ev, delta_line)
        return delta

    def image_transfer(self,
                       img,
                       h_flip=0,
                       v_flip=0,
                       transpose=0,
                       direc='forward'):
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

    def PSF_detector(self, det_res, p_x, I):
        '''
            the resolution degrades due to PSF
            I:
                the data
        '''
        M, N = I.shape
        y_axis = np.arange(-M // 2, M // 2) * p_x
        x_axis = np.arange(-N // 2, N // 2) * p_x
        XX, YY = np.meshgrid(x_axis, y_axis)

        sigma = det_res / (2 * np.sqrt(np.log(2)))
        PSF = np.exp(-(XX**2 + YY**2) / (sigma)**2)

        return np.abs(ifft2(fft2(I) * fft2(PSF)))

    def PSF_coherence(self, sigma_h, sigma_v, p_x, I):
        '''
            the resolution degrades due to temporal or spatial coherence
            sigma_h:
                horizontal convolution kernel size
            sigma_v:
                vertical convolution kernel size
            I:
                the data
        '''
        M, N = I.shape
        y_axis = np.arange(-M // 2, M // 2) * p_x
        x_axis = np.arange(-N // 2, N // 2) * p_x
        XX, YY = np.meshgrid(x_axis, y_axis)

        PSF = np.exp(-(XX**2 / sigma_h**2 + YY**2 / sigma_v**2))

        return np.abs(ifft2(fft2(I) * fft2(PSF)))

    def spherical_wavefront(self, n_size, distance, p_x, wavelength):
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
        y_axis = np.arange(-M // 2, M // 2) * p_x
        x_axis = np.arange(-N // 2, N // 2) * p_x
        XX, YY = np.meshgrid(x_axis, y_axis)

        phi_rad = k * (XX**2 / (2 * distance[1]) + YY**2 /
                                       (2 * distance[0]))
        phase_illum = np.exp(1j * phi_rad)

        return phase_illum, phi_rad

    def pattern_prop(self, I_pattern, show_fig=False):
        '''
            propagate the matched pattern to get pattern-ref image
            input:
                    I_pattern:          pattern distribution
        '''

        # energy
        # energy = 14e3

        # calculate sigma kernel size for coherence
        sigma_h = self.source_h / self.source_distance_h * self.d_propagation
        sigma_v = self.source_v / self.source_distance_v * self.d_propagation

        # I_pattern:  pattern matched,   I_img: ref image used to find the matched pattern
        I_pattern = normalize(I_pattern)

        # scale for pattern pitch to detector pixel size
        scale = self.pattern_pixel / self.p_x

        # I_pattern = rescale(I_pattern, scale)
        size_origin = I_pattern.shape
        I_pattern = cv2.resize(I_pattern, (int(I_pattern.shape[1]*scale), int(I_pattern.shape[0]*scale)), interpolation = cv2.INTER_NEAREST)
        # I_pattern = cv2.resize(I_pattern, (int(I_pattern.shape[0]*scale), int(I_pattern.shape[1]*scale)))
        # I_pattern = snd.zoom(I_pattern, scale, order=0)
        # I_pattern = np.repeat(np.repeat(I_pattern, int(scale), axis=0),
        #                       int(scale),
        #                       axis=1)
        # the pixel size after repeating expanding the matrix. Should be noted that, use nearest or linear or other interplation induces extra artifacts in the propgated pattern. So use this int pixel size for propagation and then scale the propgated pattern with the correct scales
        # p_x_prop = self.pattern_pixel / int(scale)
        p_x_prop = self.pattern_pixel / scale

        print(
            'scale pattern {} to size of P{} from size of {}, pixel size for propagation: {}'.
            format(scale, I_pattern.shape, size_origin, p_x_prop))
        # plt.imshow(I_pattern)
        # plt.show()
        # propagation
        # generate the spherical phase offset induced by the source distance.
        phase_complex, _ = self.spherical_wavefront(I_pattern.shape, [self.source_distance_v, self.source_distance_h], p_x_prop, wavelength=self.c_w)

        # A_pattern = np.sqrt((1-self.pattern_transmission)*I_pattern+self.pattern_transmission) * np.exp(1j*I_pattern*self.pattern_phase) * phase_complex
        A_pattern = np.sqrt((1-self.pattern_transmission)*I_pattern+self.pattern_transmission) * np.exp(1j*I_pattern*self.pattern_phase) 
        
        '''
            for near-field propagation, the magnification is approximated by:
                d -> d/M
                p_x -> M * p_x
            so, first propagate the wavefront for a distance of d/M, then zoom the result with a factor of M. So the final array size will be NM * NM
        '''
        M_factor = [(self.source_distance_h+self.d_propagation)/self.source_distance_h, 
                    (self.source_distance_v+self.d_propagation)/self.source_distance_v]
        d_approx = [self.d_propagation/M_factor[0], self.d_propagation/M_factor[1]]
        prColor('M factor for propagation: {}'.format(M_factor), 'green')
        prColor('equvalent distance for propagation: {}'.format(d_approx), 'green')
        size_origin = A_pattern.shape
        prColor('origin size before propagation: {}'.format(size_origin), 'green')

        N_pad = 128
        if N_pad == 0:
            # A_prop, L_out = diffraction_prop(A_pattern, p_x_prop,
            #                                  self.d_propagation, self.c_w,
            #                                  self.prop_mode)
            A_prop, L_out = prop_TF_2d(A_pattern, p_x_prop, self.c_w, d_approx)
        else:
            A_pattern_pad = np.pad(A_pattern, (N_pad, N_pad),
                                   mode='constant',
                                   constant_values=(0, 0))
            # A_prop, L_out = diffraction_prop(A_pattern_pad, p_x_prop,
            #                                  self.d_propagation, self.c_w,
            #                                  self.prop_mode)
            A_prop, L_out = prop_TF_2d(A_pattern_pad, p_x_prop, self.c_w, d_approx)

            A_prop = A_prop[N_pad:-N_pad, N_pad:-N_pad]

        I_prop = np.abs(A_prop)**2
        # zoom the diffraction pattern by M_factor
        I_prop = cv2.resize(I_prop, (int(size_origin[1]*M_factor[0]), int(size_origin[0]*M_factor[1])))
        prColor('origin size after propagation: {}'.format(I_prop.shape), 'green')

        # sys.exit()
        # # now the real pixel size is p_x_prop, need to scale it to the detector pixel size
        # scale_res = p_x_prop / self.p_x
        # # scale_res = 1

        # I_prop = snd.zoom(I_prop, scale_res, order=1)
        # print('residual scale: {}, to shape: {}'.format(
        #     scale_res, I_prop.shape))

        # crop the boundary artifact from the diffraction propagation
        # n_crop = 256
        # I_prop = I_prop[n_crop:-n_crop, n_crop:-n_crop]

        # the detector PSF influence
        I_det = self.PSF_detector(self.det_res, self.p_x, I_prop)

        I_coh = self.PSF_coherence(sigma_h, sigma_v, self.p_x, I_det)
        
        # plt.figure()
        # plt.imshow(I_prop)
        # plt.title('prop')
        # plt.figure()
        # plt.imshow(I_det)
        # plt.title('det')
        # plt.figure()
        # plt.imshow(I_coh)
        # plt.title('coh')
        # plt.show()

        return I_coh.astype(np.float32), I_det.astype(
            np.float32), I_prop.astype(np.float32)

    def img_transfer_search(self, I_img, I_pattern, result_folder):
        '''
            find the correct image transfer for the reference image, and save the results
        '''
        corr_list = []

        trans_list = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1],
                      [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        for img_transfer in trans_list:
            print('image transfer: {}'.format(img_transfer))
            pos_center, corr_math, img_small, template = self.pattern_search_coarse(
                I_img, I_pattern, img_transfer)
            corr_list.append(corr_math)

            plt.figure(figsize=(12, 5))
            plt.subplot(121)
            plt.imshow(img_small, cmap='gray')
            plt.colorbar()
            plt.title('Matched pattern')
            plt.subplot(122)
            plt.imshow(template, cmap='gray')
            plt.colorbar()
            plt.title('Raw image')
            plt.savefig(
                os.path.join(
                    result_folder,
                    'img_transfer_{}_{}_{}_center_{}x_{}y.png'.format(
                        img_transfer[0], img_transfer[1], img_transfer[2],
                        pos_center[0], pos_center[1])))
            plt.close()

        plt.figure()
        ax = plt.axes()
        plt.plot(corr_list, '-*')
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
        ax.set_xticklabels(
            ['000', '001', '010', '100', '011', '101', '110', '111'])
        plt.ylabel('correlation coefficient')
        plt.savefig(os.path.join(result_folder, 'corr_list.png'))
        plt.close()

        num_trans = corr_list.index(max(corr_list))
        prColor('max corr {} at {}'.format(corr_list[num_trans], trans_list[num_trans]), 'green')

        #TODO: save the correlation matrix, to be put in the ini

        return trans_list[num_trans]

    def pattern_search_coarse(self, I_img, I_pattern, img_transfer=[1, 0, 0]):
        # find the matched pattern position coarsely

        I_pattern = self.image_transfer(I_pattern, img_transfer[0], img_transfer[1], img_transfer[2])
        m, n = I_img.shape

        row_start = m // 2 - 100
        row_end = m // 2 + 100
        col_start = n // 2 - 100
        col_end = n // 2 + 100
        I_img_norm = normalize(I_img[row_start:row_end,
                                     col_start:col_end]) * 255
        I_img_norm = I_img_norm.astype(np.float32)

        # I_img_norm = self.image_transfer(I_img_norm, img_transfer[0], img_transfer[1], img_transfer[2])

        # find pattern matching postion, coarse searching
        meth = 'cv2.TM_CCOEFF'

        I_pattern_reshape = normalize(I_pattern.astype(np.float32)) * 255
        template = I_img_norm
        # M_factor = [(self.source_distance_h+self.d_propagation)/self.source_distance_h, 
        #             (self.source_distance_v+self.d_propagation)/self.source_distance_v]
        # # scale the pattern to find the matched position
        # I_pattern_reshape = snd.zoom(I_pattern_reshape, zoom=(M_factor[0], M_factor[1]))


        n_template_row, n_template_col = template.shape

        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(I_pattern_reshape, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + n_template_col,
                        top_left[1] + n_template_row)
        # print(top_left, bottom_right)
        img_small = I_pattern_reshape[top_left[1]:bottom_right[1],
                                      top_left[0]:bottom_right[0]]

        # corr_match = np.corrcoef(normalize_std(img_small), normalize_std(template))[0,1]
        # print('correlation coeffcient: {}'.format(corr_match))

        corr_match = np.amax(
            ssignal.correlate2d(img_small,
                                template,
                                boundary='symm',
                                mode='same'))
        print('correlation coeffcient: {}'.format(corr_match))

        x_center = int((top_left[0] + bottom_right[0]) / 2)
        y_center = int((top_left[1] + bottom_right[1]) / 2)
        print('center of pattern position: {} x; {} y'.format(
            x_center, y_center))
        return [x_center, y_center], corr_match, img_small, template

    def find_transfer_matrix(self, im1, im2):
        # use opencv to find the image transformation matrix
        # Read the images to be aligned
        # Find size of image1
        im1 = im1.astype(np.float32)
        im2 = im2.astype(np.float32)

        sz = im1.shape
        # Define the motion model
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = 5000

        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    number_of_iterations, termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(im1, im2, warp_matrix,
                                                 warp_mode, criteria)

        # assuming the affine transformation matrix as ski-image package order: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.AffineTransform

        a0 = warp_matrix[0, 0]
        a1 = warp_matrix[0, 1]
        tx = warp_matrix[0, 2]

        b0 = warp_matrix[1, 0]
        b1 = warp_matrix[1, 1]
        ty = warp_matrix[1, 2]

        rot_cita = np.arctan(b0 / a0)
        sx = 1 / (a0 / np.cos(rot_cita))
        shear = np.arctan(-a1 / b1) - rot_cita
        sy = -1/(a1 / np.sin(rot_cita + shear))

        print('sx: {}; sy: {}; rot: {}; shear: {}; tx: {}; ty: {}'.format(
            sx, sy, rot_cita / np.pi * 180, shear, tx, ty))

        # # Use warpAffine for Translation, Euclidean and Affine
        # im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        # apply scale to the image
        im2_aligned = snd.affine_transform(im2,
                                           warp_matrix,
                                           output_shape=None,
                                           output=None,
                                           order=3,
                                           mode='constant')
        # print(im2_aligned.shape)

        return warp_matrix, im2_aligned, [sx, sy], rot_cita, shear, [tx, ty]

    def pattern_search(self,
                       I_img,
                       I_pattern,
                       img_transfer):
        '''
            search the matched pattern based on the I_img distribution
            I_img:              the measured image
            I_pattern:          generated simulation image
            img_transfer:       how the measured image needs to flip, mirror or rot90 to match the simulation image
            crop:               the crop pos for the input image, use this to generated simulated detector plane image
        '''
        prColor('start to search matched pattern', 'cyan')
        pos_center, corr_math, img_small, template = self.pattern_search_coarse(
            I_img, I_pattern, img_transfer)
        warp_matrix, img_aligned, [sx, sy], rot_cita, shear, [
            tx, ty
        ] = self.find_transfer_matrix(template, img_small)

        self.scale = [sy, sx]
        self.rotation = rot_cita

        I_pattern = self.image_transfer(I_pattern, img_transfer[0], img_transfer[1],
                                    img_transfer[2])

        m, n = I_img.shape

        y0 = (pos_center[1] -int(self.det_array[0] / 2))
        y1 = (pos_center[1] -int(self.det_array[0] / 2) + self.det_array[0])
        x0 = (pos_center[0] - int(self.det_array[1] / 2))
        x1 = (pos_center[0] - int(self.det_array[1] / 2) + self.det_array[1])

        n_pad = np.amax([-y0 * (y0<=0), (y1-self.det_array[0])*(y1>self.det_array[0]),
                        -x0 * (x0<=0), (x1-self.det_array[1])*(x1>self.det_array[1])])
        if n_pad != 0:
            prColor('padding the pattern boundary to match the detector', 'green')
            I_pattern_det = np.pad(I_pattern, n_pad)
        else:
            I_pattern_det = I_pattern.copy()

        I_pattern_det = I_pattern_det[(pos_center[1] + n_pad -
                                       int(self.det_array[0] / 2)):(pos_center[1] + n_pad -
                                                    int(self.det_array[0] / 2) + self.det_array[0]),
                                      (pos_center[0] + n_pad -
                                       int(self.det_array[1] / 2)):(pos_center[0] + n_pad -
                                                    int(self.det_array[1] / 2) + self.det_array[1])]

        I_pattern_matched = I_pattern[(pos_center[1] -
                                       int(m / 2)):(pos_center[1] -
                                                    int(m / 2) + m),
                                      (pos_center[0] -
                                       int(n / 2)):(pos_center[0] -
                                                    int(n / 2) + n)]
        


        # scale and rotate back

        if self.correct_scale:
            I_pattern_matched = clipped_zoom(I_pattern_matched,
                                             (1 / sy, 1 / sx))
            I_pattern_det = clipped_zoom(I_pattern_det,
                                             (1 / sy, 1 / sx))
            prColor('correct scale', 'cyan')
        # I_pattern_matched = cv2_clipped_zoom(I_pattern_matched, zoom_factor=(1/sy, 1/sx))
        # I_pattern_matched = cv2.resize(I_pattern_matched, (n, m), fx=1/sx, fy=1/sy, interpolation = cv2.INTER_LINEAR)
        I_pattern_matched = snd.rotate(I_pattern_matched,
                                       rot_cita / np.pi * 180,
                                       reshape=False)
        I_pattern_det = snd.rotate(I_pattern_det,
                                       rot_cita / np.pi * 180,
                                       reshape=False)
        prColor('correct rotation', 'cyan')

        # I_pattern_matched = I_pattern_matched[int(I_pattern_matched.shape[0]//2-m//2): int(I_pattern_matched.shape[0]//2-m//2+m),int(I_pattern_matched.shape[1]//2-n//2): int(I_pattern_matched.shape[1]//2-n//2+n)]

        # check alignment again
        prColor('find translation and check first alignment', 'cyan')
        template = I_img[(int(m / 2) - 100):(int(m / 2) + 100),
                         (int(n / 2) - 100):(int(n / 2) + 100)]
        img_small = I_pattern_matched[(int(m / 2) - 100):(int(m / 2) + 100),
                                      (int(n / 2) - 100):(int(n / 2) + 100)]
        warp_matrix, img_aligned, [sx, sy], rot_cita, shear, [
            tx, ty
        ] = self.find_transfer_matrix(template, img_small)

        self.translation = [-ty, -tx]
        # find the translation again
        prColor('correct translation', 'cyan')
        I_pattern_matched = image_translation(I_pattern_matched, [-ty, -tx])
        I_pattern_det = image_translation(I_pattern_det, [-ty, -tx])
        # check alignment again
        template = I_img[(int(m / 2) - 100):(int(m / 2) + 100),
                         (int(n / 2) - 100):(int(n / 2) + 100)]
        img_small = I_pattern_matched[(int(m / 2) - 100):(int(m / 2) + 100),
                                      (int(n / 2) - 100):(int(n / 2) + 100)]
        prColor('check final alignment', 'cyan')
        #warp_matrix, img_aligned, [sx, sy], rot_cita, shear, [
        #    tx, ty
        #] = self.find_transfer_matrix(template, img_small)

        if self.show_alignFigure:
            plt.figure()
            plt.subplot(121)
            plt.imshow(template)
            plt.title('center of measured image')
            plt.subplot(122)
            plt.imshow(img_aligned)
            plt.title('aligned simulated image')

            plt.show()
        # estimate the source distance based on the matched pattern scales
        d_source_x = self.d_propagation / (
            (self.source_distance_h + self.d_propagation) /
            self.source_distance_h * self.scale[1] - 1)
        d_source_y = self.d_propagation / (
            (self.source_distance_v + self.d_propagation) /
            self.source_distance_v * self.scale[0] - 1)

        self.d_source_est = [d_source_y, d_source_x]
        prColor(
            'estimated source distance: {}y, {}x'.format(
                d_source_y, d_source_x), 'green')
        self.d_source_est =  [d_source_y, d_source_x]
        prColor('generating displacement offset from simulation parameters.', 'green')
        _, phase_spherical = self.spherical_wavefront(I_pattern_det.shape, [self.source_distance_v, self.source_distance_h], self.p_x, wavelength=self.c_w)
        
        displace_x_offset = np.gradient(phase_spherical/(self.p_x**2 * 2 * np.pi / self.c_w/self.d_propagation), axis=1)
        displace_y_offset = np.gradient(phase_spherical/(self.p_x**2 * 2 * np.pi / self.c_w/self.d_propagation), axis=0)

        curve_y = np.gradient(displace_y_offset, axis=0)/self.d_propagation
        curve_x = np.gradient(displace_x_offset, axis=1)/self.d_propagation

        print(1/np.mean(curve_x[100:-100, 100:-100]), 1/np.mean(curve_y[100:-100, 100:-100]))

        return I_pattern_det.astype(np.float32), displace_x_offset.astype(np.float32), displace_y_offset.astype(np.float32)

        # plt.figure()
        # plt.subplot(321)
        # plt.imshow(I_img[0:100, 0:100])
        # plt.subplot(322)
        # plt.imshow(I_pattern_matched[0:100, 0:100])

        # plt.subplot(323)
        # plt.imshow(I_img[0:100, -100:-1])

        # plt.subplot(324)
        # plt.imshow(I_pattern_matched[0:100, -100:-1])

        # plt.subplot(325)
        # plt.imshow(I_img[m//2-100: m//2+100, n//2-100:n//2+100])

        # plt.subplot(326)
        # plt.imshow(I_pattern_matched[m//2-100: m//2+100, n//2-100:n//2+100])

        # plt.show()


def normalize(v):
    return (v - np.amin(v)) / (np.amax(v) - np.amin(v))


def normalize_std(v):
    return (v - np.mean(v)) / np.std(v)


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.

    # Zooming out
    if zoom_factor[0] < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor[0]))
        zw = int(np.round(w * zoom_factor[1]))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = snd.zoom(img, zoom_factor,
                                                     **kwargs)

    # Zooming in
    elif zoom_factor[0] > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor[0]))
        zw = int(np.round(w / zoom_factor[1]))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = snd.zoom(img[top:top + zh, left:left + zw], zoom_factor,
                       **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def cv2_clipped_zoom(img, zoom_factor=0):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    ------
    Args:
        img : ndarray
            Image array
        zoom_factor : float
            amount of zoom as a ratio [0 to Inf). Default 0.
    ------
    Returns:
        result: ndarray
           numpy ndarray of the same shape of the input img zoomed by the specified factor.          
    """
    if zoom_factor[0] == 0:
        return img

    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor[0]), int(width *
                                                              zoom_factor[1])
    # print(new_height, new_width)
    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = [(bbox[0] / zoom_factor[0]).astype(np.int32),
            (bbox[1] / zoom_factor[1]).astype(np.int32),
            (bbox[2] / zoom_factor[0]).astype(np.int32),
            (bbox[3] / zoom_factor[1]).astype(np.int32)]
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height,
                                      height), min(new_width, width)
    pad_height1, pad_width1 = (height -
                               resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (
        width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2),
                (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def image_translation(img, shift):

    # # roi = lambda x: x[0:100][0:100]
    # shift, error, diffphase = register_translation(image, offset_image, 10)
    # # shift, error, diffphase = phase_cross_correlation(image, offset_image, 100)
    # # shift, error, diffphase = register_translation(roi(image), roi(offset_image), 100)

    # print('shift dist: {}, alignment error: {} and phase difference: {}'.format(shift, error, diffphase))
    # image_back = image_shift(offset_image, shift[0], shift[1])
    image_back = snd.fourier_shift(np.fft.fftn(img), shift)
    image_back = np.real(np.fft.ifftn(image_back))

    return image_back


def speckle_tracking(ref, img, para_XST, p_x, d_prop, wl, displace_offset):
    '''
        to get displacement
    '''
    # convert data type to np.float64
    ref = ref.astype(np.float32)
    img = img.astype(np.float32)
    size_origin = ref.shape

    # down-sample or not
    if para_XST['down_sampling'] != 1:
        prColor('down-sample image: {}'.format(para_XST['down_sampling']),
                'cyan')
        d_size = (int(ref.shape[1]*para_XST['down_sampling']), int(ref.shape[0]*para_XST['down_sampling']))

        # from func import binning2
        # img = binning2(img)
        # ref = binning2(ref)
        # print(img.shape)

        img = cv2.resize(img, d_size)
        ref = cv2.resize(ref, d_size)
        # img = snd.zoom(img, zoom=para_XST['down_sampling'])
        # ref = snd.zoom(ref, zoom=para_XST['down_sampling'])

    if para_XST['method'] == 'simple':
        # use opencv to find displacement roughly
        displace_x, displace_y = slop_tracking(img, ref, n_window=50)

        # down-sample or not
        if para_XST['down_sampling'] != 1:
            displace_x = cv2.resize(displace_x, (size_origin[1], size_origin[0])) * (1 / para_XST['down_sampling'])
            displace_y = cv2.resize(displace_y, (size_origin[1], size_origin[0])) * (1 / para_XST['down_sampling'])

            # displace_x = snd.zoom(
            #     displace_x, zoom=1 /
            #     para_XST['down_sampling']) * (1 / para_XST['down_sampling'])
            # displace_y = snd.zoom(
            #     displace_y, zoom=1 /
            #     para_XST['down_sampling']) * (1 / para_XST['down_sampling'])

        displace_fine = [displace_y.copy(), displace_x.copy()]

        displace_x += displace_offset[1]
        displace_y += displace_offset[0]

        if para_XST['crop_boundary'][0] != 0:
            displace_x = displace_x[
                para_XST['crop_boundary'][0]:-para_XST['crop_boundary'][0],
                para_XST['crop_boundary'][1]:-para_XST['crop_boundary'][1]]
            displace_y = displace_y[
                para_XST['crop_boundary'][0]:-para_XST['crop_boundary'][0],
                para_XST['crop_boundary'][1]:-para_XST['crop_boundary'][1]]

        displace_x -= np.mean(displace_x)
        displace_y -= np.mean(displace_y)

        DPC_y = (displace_y) * p_x / d_prop
        DPC_x = (displace_x) * p_x / d_prop

        phase = frankotchellappa(DPC_x, DPC_y) * p_x * 2 * np.pi / wl

    elif para_XST['method'] == 'WXST':
        # use WXST to find displacement accurately
        WXST_solver = WXST(img,
                           ref,
                           N_s=para_XST['template_size'],
                           cal_half_window=para_XST['window_searching'],
                           N_s_extend=4,
                           n_cores=para_XST['nCore'],
                           n_group=para_XST['nGroup'],
                           wavelet_level_cut=para_XST['wavelet_lv_cut'],
                           pyramid_level=para_XST['pyramid_level'],
                           n_iter=para_XST['n_iter'],
                           use_wavelet=para_XST['use_wavelet'],
                           use_estimate=False,
                           use_GPU=para_XST['GPU'])
        WXST_solver.run()
        displace_y, displace_x = WXST_solver.displace

        # down-sample or not
        if para_XST['down_sampling'] != 1:
            # prColor('scale 2 back', 'red')
            displace_x = cv2.resize(displace_x, (size_origin[1], size_origin[0])) * (1 / para_XST['down_sampling'])
            displace_y = cv2.resize(displace_y, (size_origin[1], size_origin[0])) * (1 / para_XST['down_sampling'])

        displace_fine = [displace_y.copy(), displace_x.copy()]

        displace_x += displace_offset[1]
        displace_y += displace_offset[0]

        if para_XST['crop_boundary'][0] != 0:
            displace_x = displace_x[
                para_XST['crop_boundary'][0]:-para_XST['crop_boundary'][0],
                para_XST['crop_boundary'][1]:-para_XST['crop_boundary'][1]]
            displace_y = displace_y[
                para_XST['crop_boundary'][0]:-para_XST['crop_boundary'][0],
                para_XST['crop_boundary'][1]:-para_XST['crop_boundary'][1]]

        displace_x -= np.mean(displace_x)
        displace_y -= np.mean(displace_y)

        DPC_y = (displace_y) * p_x / d_prop
        DPC_x = (displace_x) * p_x / d_prop

        phase = frankotchellappa(DPC_x, DPC_y) * p_x * 2 * np.pi / wl
    
    elif para_XST['method'] == 'SPINNet':
        from aps.common.measurment.beamline.wf.SPINNet_estimate import SPINNet_estimate
        # use SPINNet to find displacement accurately
        trained_folder='../SPINNet/PhaseOnly/trained_model/Result_pxShift_data_10k_T0p2_feature10_fp16_search3_longerTraining/'
        # here is the best model
        trained_model=os.path.join(trained_folder, 'training_model_002000.pt')
        setting_path=os.path.join(trained_folder, 'setting_002000.json')

        device = 'cuda' if para_XST['GPU'] else 'cpu'
        
        # # do image alignment
        # if True:
        #     pos_shift, img = image_align(ref, img)
        #     max_shift = int(np.amax(np.abs(pos_shift))+1)
        #     crop_area = lambda img: img[max_shift:-max_shift, max_shift:-max_shift]
        #     img = crop_area(img)
        #     ref = crop_area(ref)

        ref = ref / snd.uniform_filter(ref, 50) * 255
        img = img / snd.uniform_filter(img, 50) * 255

        # ref = ref / np.amax(ref) * 255
        # img = img / np.amax(img) * 255

        I_mean = np.mean(ref)
        ref = (ref - np.mean(ref)) / np.std(ref) * I_mean/2 + I_mean
        img = (img - np.mean(img)) / np.std(img) * I_mean/2 + I_mean

        I_minMax = np.amin([np.amax(ref), np.amax(img)])
        img = np.clip(img, 0, I_minMax)
        ref = np.clip(ref, 0, I_minMax)

        displace_y, displace_x = SPINNet_estimate(ref, img, trained_model, setting_path, device=device)

        # down-sample or not
        if para_XST['down_sampling'] != 1:
            # prColor('scale 2 back', 'red')
            displace_x = cv2.resize(displace_x, (size_origin[1], size_origin[0])) * (1 / para_XST['down_sampling'])
            displace_y = cv2.resize(displace_y, (size_origin[1], size_origin[0])) * (1 / para_XST['down_sampling'])

        displace_fine = [displace_y.copy(), displace_x.copy()]

        displace_x += displace_offset[1]
        displace_y += displace_offset[0]

        if para_XST['crop_boundary'][0] != 0:
            displace_x = displace_x[
                para_XST['crop_boundary'][0]:-para_XST['crop_boundary'][0],
                para_XST['crop_boundary'][1]:-para_XST['crop_boundary'][1]]
            displace_y = displace_y[
                para_XST['crop_boundary'][0]:-para_XST['crop_boundary'][0],
                para_XST['crop_boundary'][1]:-para_XST['crop_boundary'][1]]

        displace_x -= np.mean(displace_x)
        displace_y -= np.mean(displace_y)

        DPC_y = (displace_y) * p_x / d_prop
        DPC_x = (displace_x) * p_x / d_prop

        phase = frankotchellappa(DPC_x, DPC_y) * p_x * 2 * np.pi / wl
    
    elif para_XST['method'] == 'SPINNet_split':
        '''
            SPINNet_split: split the image into multiple parts and do the prediction for each part individually 
            Why: 
                After testing, it shows that the SPINNet (phase only) cannot predict too large displacement. Otherwise it will show artifact where the displacement exceeds +/-10
                So the idea is:
                1. split the image into multiple parts
                2. do the image alignment to find the overall displacement
                3. After removing the overall displacement, the predicted displacement can be limited into a small range, and then add back the overall displacement. 
                4. Finally, do the average or stitch back the parts into one displacement map. 
            
            Results:
                So now the sub-patch way shows that there is obvious boundary effect, and this seems hard to overcome. In addition, the alignment process for sub-patch is not that accurate. 
                So the idea will be just use pre-calibration to get the rough value for the curvature using the simple method. The do the pattern searching again to get the accurate value.
                

            
        '''
        from SPINNet_estimate import SPINNet_estimate
        # from func import image_align
        # use SPINNet to find displacement accurately
        trained_folder='../SPINNet/PhaseOnly/trained_model/Result_pxShift_data_10k_T0p2_feature10_fp16_search3_longerTraining/'
        # here is the best model
        trained_model=os.path.join(trained_folder, 'training_model_002000.pt')
        setting_path=os.path.join(trained_folder, 'setting_002000.json')

        device = 'cuda' if para_XST['GPU'] else 'cpu'
        
        # # do image alignment
        # if True:
        #     pos_shift, img = image_align(ref, img)
        #     max_shift = int(np.amax(np.abs(pos_shift))+1)
        #     crop_area = lambda img: img[max_shift:-max_shift, max_shift:-max_shift]
        #     img = crop_area(img)
        #     ref = crop_area(ref)

        ref = ref / snd.uniform_filter(ref, 30) * 255
        img = img / snd.uniform_filter(img, 30) * 255

        I_mean = np.mean(ref)
        ref = (ref - np.mean(ref)) / np.std(ref) * I_mean/2 + I_mean
        img = (img - np.mean(img)) / np.std(img) * I_mean/2 + I_mean

        I_minMax = np.amin([np.amax(ref), np.amax(img)])
        img = np.clip(img, 0, I_minMax)
        ref = np.clip(ref, 0, I_minMax)

        # -------------------------------- split image ---------------------------------------------------------
        from func import split_image, combine_patches
        N_split = [2, 2]
        Overlap_percent = 0.2
        raw_size, p_r_list, p_c_list, patches_img = split_image(img, N_split, overlap_percent=Overlap_percent)
        raw_size, p_r_list, p_c_list, patches_ref = split_image(ref, N_split, overlap_percent=Overlap_percent)
        # print(len(patches))
        # plt.figure()
        # for n in range(len(patches_ref)):
        #     plt.subplot(1, len(patches_ref), n+1)
        #     plt.imshow(patches_ref[n])
        # # plt.clim([0, 255])
        # plt.show()
        # -------------------------------- do the prediction for each patch -----------------------------------
        # first find the relative displacement of each patch pair
        N_patch = len(patches_img)
        displacement_patches =  []
        patches_ref_aligned = []
        for n, (p_img, p_ref) in enumerate(zip(patches_img, patches_ref)):
            # do image alignment
            # I_mean = np.mean(p_ref)
            # p_ref = (p_ref - np.mean(p_ref)) / np.std(p_ref) * I_mean/2 + I_mean
            # p_img = (p_img - np.mean(p_img)) / np.std(p_img) * I_mean/2 + I_mean
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(p_ref)
            # plt.colorbar()
            # plt.subplot(122)
            # plt.imshow(p_img)
            # plt.colorbar()
            # plt.show()
            pos_shift, p_ref = image_align(p_img, p_ref)
            displacement_patches.append(pos_shift)
            patches_ref_aligned.append(p_ref)

            prColor('relative displacement for patch {}: {}'.format(n, pos_shift), 'green')

        # do the prediction for each patch
        displace_y_list = []
        displace_x_list = []

        for n, (p_img, p_ref) in enumerate(zip(patches_img, patches_ref_aligned)):
            displace_y_patch, displace_x_patch = SPINNet_estimate(p_ref, p_img, trained_model, setting_path, device=device)
            print(displacement_patches[n])
            displace_y_list.append(displace_y_patch - displacement_patches[n][0])
            displace_x_list.append(displace_x_patch - displacement_patches[n][1])

        displace_y = combine_patches(displace_y_list, p_r_list, p_c_list, raw_size)
        displace_x = combine_patches(displace_x_list, p_r_list, p_c_list, raw_size)

        # down-sample or not
        if para_XST['down_sampling'] != 1:
            # prColor('scale 2 back', 'red')
            displace_x = cv2.resize(displace_x, (size_origin[1], size_origin[0])) * (1 / para_XST['down_sampling'])
            displace_y = cv2.resize(displace_y, (size_origin[1], size_origin[0])) * (1 / para_XST['down_sampling'])
     
        displace_fine = [displace_y.copy(), displace_x.copy()]
        displace_x += displace_offset[1]
        displace_y += displace_offset[0]

        if para_XST['crop_boundary'][0] != 0:
            displace_x = displace_x[
                para_XST['crop_boundary'][0]:-para_XST['crop_boundary'][0],
                para_XST['crop_boundary'][1]:-para_XST['crop_boundary'][1]]
            displace_y = displace_y[
                para_XST['crop_boundary'][0]:-para_XST['crop_boundary'][0],
                para_XST['crop_boundary'][1]:-para_XST['crop_boundary'][1]]

        displace_x -= np.mean(displace_x)
        displace_y -= np.mean(displace_y)

        DPC_y = (displace_y) * p_x / d_prop
        DPC_x = (displace_x) * p_x / d_prop

        phase = frankotchellappa(DPC_x, DPC_y) * p_x * 2 * np.pi / wl

    return displace_y, displace_x, DPC_y, DPC_x, phase, displace_fine


def get_local_curvature(displace_y, displace_x, d_prop):
    '''
        get local curvature from the differential phase
        phi
    '''
    return np.gradient(displace_y, axis=0)/d_prop, np.gradient(displace_x, axis=1)/d_prop 

def do_recal_d_source(I_img_raw, I_img, para_pattern, pattern_find, image_transfer_matrix, boundary_crop, para_XST, para_simulation, method='simple_speckle'):
    c_w = pattern_find.c_w
    para_XST_simple = para_XST.copy()
    para_XST_simple['method'] = 'simple'
    para_XST_simple['down_sampling'] = 0.5
    if para_pattern['propagated_pattern'] == 'None':
        prColor('MESSAGE: pattern image,  ' + para_pattern['pattern_path'], 'green')
        I_pattern = np.load(para_pattern['pattern_path']).astype(np.float32)
        I_pattern = (1 - I_pattern)

        # propagate the pattern to the detector
        prColor('generating simulated pattern...', 'cyan')
        I_coh, _, _ = pattern_find.pattern_prop(I_pattern)

    if para_pattern['propagated_patternDet'] == 'None':
        # use central part of the raw image to generate the simulated detector reference image
        center_crop = lambda img: img[int(img.shape[0] // 2 - 256):int(img.shape[0] // 2 +256), int(img.shape[1] // 2 - 256):int(img.shape[1] // 2 + 256)]

        I_img_central = center_crop(I_img_raw)

        if image_transfer_matrix is None:
            # find the proper image transfer for the reference image which matches the pattern distribution
            image_transfer_matrix = pattern_find.img_transfer_search(I_img_central, I_coh, result_folder)
            I_simu_whole, displace_x_offset, displace_y_offset = pattern_find.pattern_search(I_img_central, I_coh, image_transfer_matrix)
        else:
            I_simu_whole, displace_x_offset, displace_y_offset = pattern_find.pattern_search(I_img_central, I_coh, image_transfer_matrix)

        with open(os.path.join(para_pattern['saving_path'], "image_transfer_matrix.npy"), 'wb') as f: np.save(f, np.array(image_transfer_matrix), allow_pickle=False)

    if method == 'geometric':
        d_source_v, d_source_h = pattern_find.d_source_est                                          
        prColor('re-calculated source distance: {}y    {}x'.format(d_source_v, d_source_h), 'cyan')

        return [d_source_v, d_source_h]
    elif method == 'simple_speckle':
        I_simu = boundary_crop(I_simu_whole)
        displace_y_offset = boundary_crop(displace_y_offset)
        displace_x_offset = boundary_crop(displace_x_offset)
        displace_y_offset = displace_y_offset - np.mean(displace_y_offset)
        displace_x_offset = displace_x_offset - np.mean(displace_x_offset)

        I_img = normalize(I_img) * 255
        I_simu = normalize(I_simu) * 255

        prColor('speckle tracking mode: area. Will use the whole cropping area for calculation.', 'cyan')
        displace_y, displace_x, _, _, _, _ = speckle_tracking(
            I_simu, I_img, para_XST_simple, para_simulation['p_x'],
            para_simulation['d_prop'], c_w, displace_offset=[displace_y_offset, displace_x_offset])

        # # do filter for displacement before calcuating the curvature
        # displace_x_filtered = snd.gaussian_filter(displace_x, 21)
        # displace_y_filtered = snd.gaussian_filter(displace_y, 21)

        curve_y, curve_x = get_local_curvature(displace_y, displace_x, para_simulation['d_prop'])
        prColor('re-calculated source distance: {}y    {}x'.format(1/np.mean(curve_y), 1/np.mean(curve_x)), 'cyan')

        return [1/np.mean(curve_y), 1/np.mean(curve_x)]
    else:
        prColor('Wrong method for source distance re-calculation', 'red')

if __name__ == "__main__":
    # paremater settings
    parser = argparse.ArgumentParser(
        description='experimental data analysis for absolute phase measurement',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # shared args
    # ============================================================
    parser.add_argument('--img',
                        type=str,
                        default='./images/1DCRL/sample_00001.tif',
                        help='path to sample image')
    parser.add_argument('--dark',
                        type=str,
                        default='None',
                        help='file path to the dark image')
    parser.add_argument('--flat',
                        type=str,
                        default='None',
                        help='file path to the flat image')

    parser.add_argument('--result_folder',
                        type=str,
                        default='./images/1DCRL/results',
                        help='saving folder')
    parser.add_argument('--pattern_path',
                        type=str,
                        default='./mask/RanMask5umB0.npy',
                        help='path to mask design pattern')
    parser.add_argument('--propagated_pattern',
                        type=str,
                        default='./images/1DCRL/propagated_pattern.npz',
                        help='if None, will create one in the data folder')
    parser.add_argument('--propagated_patternDet',
                        type=str,
                        default='./images/1DCRL/propagated_patternDet.npz',
                        help='if None, will search from the propagated pattern. Its size is determined by the det_size')

    parser.add_argument('--estimation_method',
                        type=str,
                        default='simple_speckle',
                        help='simple_speckle or geometric: simple_speckle means using the slope_tracking to estimate the overall source distance;\n' + \
                             'geometric means using the image scalling factor to get the overall source distance')

    parser.add_argument(
        '--saving_path',
        type=str,
        default=None,
        help='if None, will save the propagated pattern file to the data folder'
    )

    parser.add_argument(
        "--crop",
        nargs="+",
        type=int,
        default=[450, 1000, 500, 1000],
        help=
        'image crop, if is [256], central crop. if len()==4, boundary crop, if is 0, use gui crop, if is -1, use auto-crop'
    )

    # parameters for generating simulated pattern
    parser.add_argument(
        "--img_transfer_matrix",
        nargs="+",
        type=int,
        default=[1, 0, 0],
        help=
        'the image transfer matrix to make the images match with the simulated pattern.'
    )
    parser.add_argument('--find_transferMatrix',
                        default=False,
                        action='store_true',
                        help='search the image transfer matrix or not')
    parser.add_argument('--det_size',
                        nargs="+",
                        type=int,
                        default=[2160, 2560],
                        help=
                        'detector array size, need to be same with the collected image')
    parser.add_argument('--p_x',
                        default=0.65e-6,
                        type=float,
                        help='pixel size')
    parser.add_argument('--energy',
                        default=20e3,
                        type=float,
                        help='X-ray energy')
    parser.add_argument('--pattern_size',
                        default=4.985-6,
                        type=float,
                        help='mask pattern design pixel size') 
    parser.add_argument('--pattern_thickness',
                        default=1.5e-6,
                        type=float,
                        help='mask pattern thickness')
    parser.add_argument('--pattern_T',
                        default=0.613,
                        type=float,
                        help='mask pattern transmission')
    parser.add_argument('--d_prop',
                        default=462e-3,
                        type=float,
                        help='detector to mask distance')
    parser.add_argument('--d_source_v',
                        default=60,
                        type=float,
                        help='vertical source distance')   
    parser.add_argument('--d_source_h',
                        default=60,
                        type=float,
                        help='horizontal source distance')        
    parser.add_argument('--source_v',
                        default=10e-6,
                        type=float,
                        help='vertical source size')   
    parser.add_argument('--source_h',
                        default=277e-6,
                        type=float,
                        help='horizontal source size')     
    parser.add_argument('--det_res',
                        default=1.5e-6,
                        type=float,
                        help='detector spatial resolution')
    parser.add_argument('--correct_scale',
                        default=False,
                        action='store_true',
                        help='correct mask pattern scales or not. default is False. This will remove the parabolic wavefront in the simulated pattern')
    parser.add_argument('--show_alignFigure',
                        default=False,
                        action='store_true',
                        help='show aligned figure or not')
                        
    parser.add_argument('--d_source_recal',
                        default=False,
                        action='store_true',
                        help='recalculate the source distance or not. If so, will use the simple method to recalculate the source distance.')

    parser.add_argument('--propagator',
                        default='RS',
                        type=str,
                        help='propagation method for near-field diffraction. Default is RS.')

    # parameters for WXST calculation

    parser.add_argument(
                        "--mode",
                        default='area',
                        type=str,
                        help=
                        'mode for speckle tracking. area: whole crop area; centralLine: vertical and horizontal central line with a width of parser.lineWidth;'
    )
    parser.add_argument('--lineWidth', type=int, default=5, help='line width to calculate the speckle tracking in centralLine mode. The unit is pattern size. Means that 5 is actually 5*pattern_size, such as 25um width')
    parser.add_argument('--lineDirection', type=str, default='b', help='direction to calculate the speckle tracking in centralLine mode: (v)ertical, (h)orizontal, (b)oth')

    parser.add_argument('--down_sampling', type=float, default=1, help='down-sample images to reduce memory cost and accelerate speed.')
    parser.add_argument('--crop_boundary', type=int, default=-1, help='crop the differential phase boundary. -1 will use the searching window. 0 means no cropping')
    parser.add_argument('--method',
                        default='WXST',
                        type=str,
                        help='speckle tracking method. simple: slope-tracking, fast but less accurate; WXST: wavelet speckle tracking.')
    parser.add_argument('--GPU',
                        default=False,
                        action='store_true',
                        help='Use GPU or not. GPU can be 2 times faster. But multi-resolution process is disabled.')
    parser.add_argument('--use_wavelet',
                        default=False,
                        action='store_true',
                        help='use wavelet transform or not.')
    parser.add_argument('--wavelet_lv_cut',
                        default=2,
                        type=int,
                        help='wavelet cutting level')
    parser.add_argument('--pyramid_level',
                        default=1,
                        type=int,
                        help='pyramid level used for speckle tracking.')
    parser.add_argument('--n_iter',
                        default=1,
                        type=int,
                        help='number of iteration for speckle tracking. 1 is good.')
    parser.add_argument('--template_size',
                        default=11,
                        type=int,
                        help='template size in the WXST')
    parser.add_argument('--window_searching',
                        default=10,
                        type=int,
                        help='searching window of speckle tracking. Means the largest displacement can be calculated.')
    parser.add_argument('--nCores',
                        default=8,
                        type=int,
                        help='number of CPU cores used for calculation.')
    parser.add_argument('--nGroup',
                        default=1,
                        type=int,
                        help='number of groups that parallel calculation is splitted into.')

    parser.add_argument('--verbose',
                        default=0,
                        type=int,
                        help='verbose yes(1)/no(0)')
    parser.add_argument('--simple_analysis',
                        default=0,
                        type=int,
                        help='simple analysis yes(1)/no(0)')

    args = parser.parse_args()

    file_img    = args.img
    file_folder = os.path.dirname(args.img)
  
    result_folder = args.result_folder
    if not os.path.exists(result_folder): os.makedirs(result_folder)
    
    para_pattern = {
        'pattern_path': args.pattern_path,  # path to raw binary pattern file
        'propagated_pattern': args.propagated_pattern,  # load saved propagated pattern or not, if None, will calculate it and save it
        'estimation_method': args.estimation_method,
        'saving_path': file_folder if args.saving_path is None else args.saving_path,  #if propagated_pattern is None, save the simulated to this path
        'propagated_patternDet': args.propagated_patternDet, # propagated transformed simulated reference image at detector, if None, will search from the propagated pattern.
    }

    # roi_img = [650, 1380, 550, 2000]
    # roi_img = [750, 1280, 1050, 1500]
    # roi_img = [450, 1000, 500, 1000]

    para_simulation = {
        'p_x': args.p_x,  # detector pixel size
        'pattern_size': args.pattern_size,  #4.985e-6,       # mask pitch size
        'pattern_T': args.pattern_T,  # mask transmission
        'energy': args.energy,  # energy
        'pattern_thickness': args.pattern_thickness,  # mask thickness
        'd_prop': args.d_prop,  # mask to detector distance
        # the source distance needs to be relative large so there's no artifact in the diffraction propagation due to improper p_x and pattern size. For example, 60 meters is good.
        'd_sv': args.d_source_v,  # vertical source distance
        'd_sh': args.d_source_h,  # horizontal source distance
        'sv': args.source_v,  # vertical source size
        'sh': args.source_h,  # horizontal source size
        'det_res': args.det_res,  # detector resolution
        'det_size': args.det_size, # detector array size, will save the same shape simulated reference in the propagated_pattern's folder
        'propagator': args.propagator,  # propagator for near-field diffraction
        'correct_scale': args.correct_scale,  # if correct horizontal and vertical scales
        'showAlignFigure': args.show_alignFigure, # if show aligned figure.
        'd_source_recal':   args.d_source_recal, # re-calculate the source distance or not, if so, use simple method to get the new source distance
    }

    para_XST = {
        'down_sampling': args.down_sampling,  # down-sample to reduce calculation cost, [0~1]
        'crop_boundary': [args.window_searching + args.template_size*int(1/args.down_sampling), args.window_searching + args.template_size*int(1/args.down_sampling)] if args.crop_boundary == -1 else [args.crop_boundary, args.crop_boundary],   # crop boundary of dx and dy.
        'method':
        args.method,  # method to get displacement, simple: slope-tracking, fast,less accurate; WXST
        'GPU': args.GPU,  # use GPU for WXST or not
        'template_size': args.template_size,  # template size, half window
        'window_searching': int(args.window_searching*args.down_sampling),  # searching window size, half window
        'nCore': args.nCores,  # number of cores used
        'nGroup': args.nGroup,  # number of parallel data group
        'use_wavelet': args.use_wavelet,  # use wavelet or not
        'wavelet_lv_cut': args.wavelet_lv_cut,  #wavelet cutting level
        'pyramid_level': args.pyramid_level,  # pyramid level
        'n_iter': args.n_iter,  # number of iter for repeating calculation
    }

    # image transfer to match the pattern and reference image, if None, will automatically search the transfer matrix
    image_transfer_matrix = None if args.find_transferMatrix else args.img_transfer_matrix

    if args.verbose == 1: blockPrint()

    # =====================  start to find the pattern   ================================================

    I_img_raw = load_image(file_img)

    if args.dark == 'None':
        dark = np.zeros(I_img_raw.shape)
    else:
        dark = load_image(args.dark)

    if args.flat == 'None':
        # flat = 1
        flat = snd.uniform_filter(I_img_raw, size=5*(args.pattern_size/args.p_x))
    else:
        flat = load_image(args.flat)

    if not args.simple_analysis == 2:
        if len(args.crop) == 4:
            # boundary crop, use the corner index [y0, y1, x0, x1]
            # boundary_crop = lambda img: img[int(args.crop[0]):int(args.crop[1]),
            #                                 int(args.crop[2]):int(args.crop[3])]
            pass
        elif len(args.crop) == 1:
            if args.crop[0] == 0:
                # use gui crop
                print("before crop------------------------------------------------")
                _, corner = crop_gui(I_img_raw)
                print("after crop------------------------------------------------")

                args.crop = [
                    int(corner[0][0]),
                    int(corner[1][0]),
                    int(corner[0][1]),
                    int(corner[1][1])
                ]
            elif args.crop[0] == -1:
                # use auto-crop according to the intensity boundary. rectangular shapess
                args.crop = auto_crop(flat, shrink=0.85)
            else:
                # central crop
                corner = [int(I_img_raw.shape[0] // 2 - args.crop[0] // 2),
                        int(I_img_raw.shape[0] // 2 + args.crop[0] // 2),
                        int(I_img_raw.shape[1] // 2 - args.crop[0] // 2),
                        int(I_img_raw.shape[1] // 2 + args.crop[0] // 2),
                    ]
                args.crop = corner
        else:
            # error input
            prColor(
                'error: wrong crop option. 0 for gui crop; [256] for central crop; [y0, y1, x0, x1] for bournday crop',
                'red')
            sys.exit()

    for key, value in args.__dict__.items(): prColor('{}: {}'.format(key, value), 'cyan')
    write_json(args.result_folder, 'setting', args.__dict__)

    if args.simple_analysis > 0:
        with open(os.path.join(args.result_folder, "raw_image.npy"), 'wb') as f: np.save(f, I_img_raw, allow_pickle=False)
        sys.exit(0)

    # for the boundary, extend the cropping area by search_window+template_size
    extend_boundary = args.window_searching + args.template_size*int(1/args.down_sampling)
    boundary_crop = lambda img: img[int(args.crop[0]-extend_boundary):int(args.crop[1]+extend_boundary),
                                        int(args.crop[2]-extend_boundary):int(args.crop[3]+extend_boundary)]

    I_img = boundary_crop(I_img_raw)
    I_img_raw = (I_img_raw - dark) / (flat - dark)
    flat  = boundary_crop(flat)
    dark  = boundary_crop(dark)
    I_img = (I_img - dark) / (flat - dark)

    # to find the pattern from the reference image
    pattern_find = pattern_search(ini_para=para_simulation)

    # -------------------------------- do the re-calculation of source distance -------------------------------------
    if args.d_source_recal and para_pattern['propagated_pattern'] == 'None' and para_pattern['propagated_patternDet'] == 'None':
        prColor('Re-calculate the source distance according to the current value', 'cyan')
        # estimation method, simple_speckle or geometric, simple_speckle means using the slope_tracking to estimate the overall source distance; geometric means using the image scalling factor to get the overall source distance
        est_method = para_pattern['estimation_method']
        d_source_recal = do_recal_d_source(I_img_raw, I_img, para_pattern, pattern_find, image_transfer_matrix, boundary_crop, para_XST, para_simulation, method=est_method)
    
        prColor('use the recalculated source distance to re-generate the matched pattern', 'light_gray')

        para_simulation['d_sv_ini'] = para_simulation['d_sv']
        para_simulation['d_sh_ini'] = para_simulation['d_sh']
        
        para_simulation['d_sv'] = d_source_recal[0]
        para_simulation['d_sh'] = d_source_recal[1]
    else:
        para_simulation['d_sv_ini'] = para_simulation['d_sv']
        para_simulation['d_sh_ini'] = para_simulation['d_sh']

    print('change source distance to:', para_simulation['d_sv'], para_simulation['d_sh'])

    # to find the pattern from the reference image
    pattern_find = pattern_search(ini_para=para_simulation)
    if para_pattern['propagated_pattern'] == 'None':
        prColor('MESSAGE: pattern image,  ' + para_pattern['pattern_path'],
                'green')
        I_pattern = np.load(para_pattern['pattern_path']).astype(np.float32)
        # I_pattern = I_pattern
        I_pattern = (1 - I_pattern)

        # propagate the pattern to the detector
        prColor('generating simulated pattern...', 'cyan')
        I_coh, I_det, I_prop = pattern_find.pattern_prop(I_pattern)

        np.savez(os.path.join(para_pattern['saving_path'], 'propagated_pattern.npz'), I_coh=I_coh)
    else:
        if para_pattern['propagated_patternDet'] == 'None':
            # load the pattern from the saved file
            prColor('MESSAGE: load propagated pattern,  ' + para_pattern['propagated_pattern'], 'green')
            data_content = np.load(para_pattern['propagated_pattern'])
            I_coh = data_content['I_coh']
            # I_det = data_content['I_det']
            # I_prop = data_content['I_prop']

    if para_pattern['propagated_patternDet'] == 'None':
        # use central part of the raw image to generate the simulated detector reference image
        center_crop = lambda img: img[int(img.shape[0] // 2 - 256):int(img.shape[0] // 2 +256),
                                          int(img.shape[1] // 2 - 256):int(img.shape[1] // 2 + 256)]
        I_img_central = center_crop(I_img_raw)

        if image_transfer_matrix is None:
            # find the proper image transfer for the reference image which matches the pattern distribution
            image_transfer_matrix = pattern_find.img_transfer_search(I_img_central, I_coh, result_folder)
            I_simu_whole, displace_x_offset, displace_y_offset = pattern_find.pattern_search(I_img_central, I_coh, image_transfer_matrix)
        else:
            I_simu_whole, displace_x_offset, displace_y_offset = pattern_find.pattern_search(I_img_central, I_coh, image_transfer_matrix)

        with open(os.path.join(para_pattern['saving_path'], "image_transfer_matrix.npy"), 'wb') as f: np.save(f, np.array(image_transfer_matrix), allow_pickle=False)

        np.savez(os.path.join(para_pattern['saving_path'], 'propagated_patternDet.npz'),
                 I_simu_whole=I_simu_whole,
                 displace_x_offset=displace_x_offset,
                 displace_y_offset=displace_y_offset)

    else:
        # load the simulatd pattern from the saved file
        prColor(
            'MESSAGE: load propagated pattern at detector plane,  ' +
            para_pattern['propagated_patternDet'], 'green')
        data_content = np.load(para_pattern['propagated_patternDet'])
        I_simu_whole = data_content['I_simu_whole']
        displace_x_offset = data_content['displace_x_offset']
        displace_y_offset = data_content['displace_y_offset']

    
    I_simu = boundary_crop(I_simu_whole)
    displace_y_offset = boundary_crop(displace_y_offset)
    displace_x_offset = boundary_crop(displace_x_offset)
    displace_y_offset = displace_y_offset - np.mean(displace_y_offset)
    displace_x_offset = displace_x_offset - np.mean(displace_x_offset)


    # plt.figure(figsize=(10,5))
    # plt.subplot(121)
    # plt.imshow(displace_x_offset)
    # plt.colorbar()
    # plt.subplot(122)
    # plt.imshow(displace_y_offset)
    # plt.colorbar()
    # plt.show()
    # start to estimate the displacement from the simulated image and the measured image

    I_img = normalize(I_img) * 255
    I_simu = normalize(I_simu) * 255

    # -------------------------------- do alignment ----------------------------------------------
    if True:
        # do image alignment
        pos_shift, I_simu = image_align(I_img, I_simu)
        max_shift = int(np.amax(np.abs(pos_shift)) + 1)

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(I_img)
    # plt.subplot(122)
    # plt.imshow(I_simu)
    # plt.show()
    # choose the proper speckle tracking mode, either area or centralLine
    c_w = pattern_find.c_w
    if args.mode == 'area':
        prColor('speckle tracking mode: area. Will use the whole cropping area for calculation.', 'cyan')
        displace_y, displace_x, DPC_y, DPC_x, phase, displace_fine = speckle_tracking(
            I_simu, I_img, para_XST, para_simulation['p_x'],
            para_simulation['d_prop'], c_w, displace_offset=[displace_y_offset, displace_x_offset])
        
        block_width = int(args.lineWidth*args.pattern_size / args.p_x) + 2 * para_XST['window_searching']

        # do filter for displacement before calcuating the curvature
        # displace_x_filtered = snd.gaussian_filter(displace_x, 21)
        # displace_y_filtered = snd.gaussian_filter(displace_y, 21)

        line_displace_y = displace_y[:, int(I_img.shape[0] // 2 - block_width // 2):int(I_img.shape[0] // 2 - block_width // 2 + block_width)]
        line_displace_x = displace_x[int(I_img.shape[0] // 2 - block_width // 2):int(I_img.shape[0] // 2 - block_width // 2 + block_width), :]

        line_displace = [np.mean(line_displace_y, axis=1), np.mean(line_displace_x, axis=0)]
        line_displace = [line_displace[0] - np.mean(line_displace[0]), line_displace[1] - np.mean(line_displace[1])]

        line_dpc = [line_displace[0] * para_simulation['p_x'] / para_simulation['d_prop'], 
                    line_displace[1] * para_simulation['p_x'] / para_simulation['d_prop']]
        line_phase = [np.cumsum(line_dpc[0])*para_simulation['p_x'] * 2 * np.pi / c_w,
                     np.cumsum(line_dpc[1])*para_simulation['p_x'] * 2 * np.pi / c_w]
        line_curve = [np.gradient(line_displace[0])/para_simulation['d_prop'],
                            np.gradient(line_displace[1])/para_simulation['d_prop']]

        curve_y, curve_x = get_local_curvature(displace_y, displace_x, para_simulation['d_prop'])
        prColor('mean curvature: {}y    {}x'.format(1/np.mean(curve_y), 1/np.mean(curve_x)), 'cyan')

        plt.figure(figsize=(10,8))
        plt.subplot(221)
        plt.imshow(displace_fine[0])
        plt.colorbar()
        plt.title('fine displace y')
        plt.subplot(222)
        plt.imshow(displace_fine[1])
        plt.colorbar()
        plt.title('fine displace x')
        plt.subplot(223)
        plt.imshow(displace_y)
        plt.colorbar()
        plt.title('displace y')
        plt.subplot(224)
        plt.imshow(displace_x)
        plt.colorbar()
        plt.title('displace x')
        plt.savefig(os.path.join(args.result_folder, 'displace_fine.png'), dpi=150)
        plt.close()

        line_curve_filter = [snd.gaussian_filter(line_curve[0], 21), snd.gaussian_filter(line_curve[1], 21)]
        # line_curve_filter = [line_curve[0], line_curve[1]]
        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.plot(line_curve[0], 'k')
        plt.plot(line_curve_filter[0], 'r')
        plt.xlabel('[px]')
        plt.ylabel('[1/m]')
        plt.grid()
        plt.title('vertical curvature')
        plt.subplot(122)
        plt.plot(line_curve[1], 'k')
        plt.plot(line_curve_filter[1], 'r')
        plt.xlabel('[px]')
        plt.ylabel('[1/m]')
        plt.grid()
        plt.title('horizontal curvature')
        plt.savefig(os.path.join(args.result_folder, 'linecurve_filter.png'), dpi=150)
        plt.close()

        write_json(args.result_folder, 'result', {'avg_curvature_x': 1/np.mean(line_curve[1]),
                                                    'avg_curvature_y': 1/np.mean(line_curve[0])})
        # To do: saving data and figures. Get 1D line profile and curvature profile.
        save_figure(image_pair=[['displace_x', displace_x, '[px]'],
                                ['displace_y', displace_y, '[px]'],
                                ['curve_y', curve_y, '[1/m]'],
                                ['curve_x', curve_x, '[1/m]'],
                                ['phase', phase, '[rad]'],
                                ['flat', flat, 'intensity'],
                                ['displace_x_fine', displace_fine[1], '[px]'],
                                ['displace_y_fine', displace_fine[0], '[px]']], path=args.result_folder, p_x=args.p_x, extention='.png')

        save_figure_1D(image_pair=[['line_displace_x', line_displace[1], '[px]'],
                                ['line_phase_x', line_phase[1], '[rad]'],
                                ['line_displace_y', line_displace[0], '[px]'],
                                ['line_phase_y', line_phase[0], '[rad]'],
                                ['line_curve_y', line_curve_filter[0], '[1/m]'],
                                ['line_curve_x', line_curve_filter[1], '[1/m]']], path=args.result_folder, p_x=args.p_x)
        
        save_data({'displace_x': displace_x, 'displace_y': displace_y, 'phase': phase, 
                    'line_phase_y': line_phase[0], 'line_displace_y': line_displace[0], 'line_curve_y': line_curve_filter[0], 'line_phase_x': line_phase[1], 'line_displace_x': line_displace[1], 'line_curve_x': line_curve_filter[1]}, args.result_folder, args.p_x)

    elif args.mode == 'centralLine':
        prColor('speckle tracking mode: centralLine. Will use the central linewidth of {}um for calculation.'.format(args.lineWidth*args.pattern_size*1e6), 'cyan')
        # crop the vertical and horizontal block for calculation
        block_width = int(args.lineWidth*args.pattern_size / args.p_x) + 2 * (args.window_searching + args.template_size*int(1/args.down_sampling))

        image_pair   = []
        result       = {}
        data         = {}
        avg_source_d = [np.nan, np.nan]

        def process_centralLine(index=0):
            I_img_i = I_img[:, int(I_img.shape[index] // 2 - block_width // 2):int(I_img.shape[index] // 2 - block_width // 2 + block_width)]
            I_simu_i = I_simu[:, int(I_img.shape[index] // 2 - block_width // 2):int(I_img.shape[index] // 2 - block_width // 2 + block_width)]
            displace_y_offset_i = displace_y_offset[:, int(I_img.shape[index] // 2 - block_width // 2):int(I_img.shape[index] // 2 - block_width // 2 + block_width)]
            displace_x_offset_i = displace_x_offset[:, int(I_img.shape[index] // 2 - block_width // 2):int(I_img.shape[index] // 2 - block_width // 2 + block_width)]

            displace_i = [None, None]
            displace_i[0], displace_i[1], _, _, _, _ = speckle_tracking(
                I_simu_i, I_img_i, para_XST, para_simulation['p_x'],
                para_simulation['d_prop'], c_w, displace_offset=[displace_y_offset_i, displace_x_offset_i])

            line_displace = np.mean(displace_i[index], axis=1 if index==0 else 0)
            line_displace = line_displace - np.mean(line_displace)

            # get phase and curveature for central line profile
            line_dpc   = line_displace * para_simulation['p_x'] / para_simulation['d_prop']
            line_phase = np.cumsum(line_dpc)*para_simulation['p_x'] * 2 * np.pi / c_w
            line_curve = np.gradient(line_displace)/para_simulation['d_prop']
            avg_s_d    = 1/np.mean(line_curve)
            # filter the line curve
            line_curve_filter = snd.gaussian_filter(line_curve, 21)

            suffix = 'y' if index==0 else 'x'
            image_pair.append(['line_displace_' + suffix, line_displace,     '[px]'])
            image_pair.append(['line_phase_' + suffix,    line_phase,        '[rad]'])
            image_pair.append(['line_curve_' + suffix,    line_curve_filter, '[1/m]'])
            result['avg_source_d_' + suffix] = avg_s_d
            data['line_displace_' + suffix]  = line_displace
            data['line_phase_' + suffix   ]  = line_phase
            data['line_curve_' + suffix   ]  = line_curve_filter
            avg_source_d[index]              = avg_s_d

        if args.lineDirection == 'v' or args.lineDirection == 'b': process_centralLine(0)
        if args.lineDirection == 'h' or args.lineDirection == 'b': process_centralLine(1)

        prColor('mean source distance: {}y    {}x'.format(avg_source_d[0], avg_source_d[1]), 'cyan')
        save_figure_1D(image_pair=image_pair, path=args.result_folder, p_x=args.p_x)
        write_json(args.result_folder, 'result', result)
        save_data(data, args.result_folder, args.p_x)

