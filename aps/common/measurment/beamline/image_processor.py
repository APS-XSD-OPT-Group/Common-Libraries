#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2022, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2022. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# ----------------------------------------------------------------------- #
import os
import glob
import random
import threading
import time
import pathlib
import numpy

from aps.common.measurment.beamline.image_collector import FILE_NAME_PREFIX
from aps.common.measurment.beamline.wf import SCRIPT_DIRECTORY

PIXEL_SIZE           = 0.65e-6
IMAGE_SIZE_PIXEL_HxV = [2160, 2560]

class ImageProcessor():
    def __init__(self,
                 data_collection_directory,
                 file_name_prefix=FILE_NAME_PREFIX,
                 simulated_mask_directory=None,
                 energy=20000.0,
                 source_distance=[1.5, 1.5],
                 image_transfer_matrix=[0, 1, 0]
                 ):
        self.__data_collection_directory = data_collection_directory
        self.__file_name_prefix          = file_name_prefix
        self.__simulated_mask_directory  = simulated_mask_directory
        self.__energy                    = energy
        self.__source_distance           = source_distance
        self.__image_transfer_matrix     = image_transfer_matrix


    def generate_simulated_mask(self, image_index_for_mask=1, verbose=False):
        self.__image_transfer_matrix, is_new_mask = _generate_simulated_mask(data_collection_directory=self.__data_collection_directory,
                                                                             file_name_prefix=self.__file_name_prefix,
                                                                             mask_directory=self.__simulated_mask_directory,
                                                                             energy=self.__energy,
                                                                             source_distance=self.__source_distance,
                                                                             image_index=image_index_for_mask,
                                                                             verbose=verbose)
        return self.__image_transfer_matrix, is_new_mask

    def get_image_data(self, image_index, verbose=False):
        return _get_image_data(self.__data_collection_directory,
                               self.__file_name_prefix,
                               self.__simulated_mask_directory,
                               self.__energy,
                               self.__source_distance,
                               self.__image_transfer_matrix,
                               image_index=image_index,
                               verbose=verbose)

    def process_image(self, image_index, verbose=False):
        _process_image(self.__data_collection_directory,
                       self.__file_name_prefix,
                       self.__simulated_mask_directory,
                       self.__energy,
                       self.__source_distance,
                       self.__image_transfer_matrix,
                       image_index=image_index,
                       verbose=verbose)

    def process_images(self, verbose=False):
        for file in os.listdir(self.__data_collection_directory):
            if pathlib.Path(file).suffix == ".tif" and self.__file_name_prefix in file:
                self.process_image(image_index=int(file.split('.tif')[0][-5:]), verbose=verbose)

    def start_processor_monitor(self, n_threads, verbose=False):
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(1)

        self.__active_threads = [None] * n_threads

        for i in range(n_threads):
            self.__active_threads[i] = ProcessingThread(thread_id = i+1,
                                                        data_collection_directory=self.__data_collection_directory,
                                                        file_name_prefix=self.__file_name_prefix,
                                                        simulated_mask_directory=self.__simulated_mask_directory,
                                                        energy=self.__energy,
                                                        source_distance=self.__source_distance,
                                                        image_transfer_matrix=self.__image_transfer_matrix,
                                                        verbose=verbose
                                                        )
            self.__active_threads[i].start()

    def wait_process_to_end(self):
        active = True
        time.sleep(1)
        n_threads = len(self.__active_threads)
        status = numpy.full(n_threads, False)

        while(active):
            for i in range(n_threads): status[i] = self.__active_threads[i].is_alive()
            active = numpy.any(status, where=status==True)

            if active: time.sleep(1)

class ProcessingThread(threading.Thread):

    def __init__(self, thread_id, data_collection_directory, file_name_prefix, simulated_mask_directory, energy, source_distance, image_transfer_matrix, verbose=False):
        super(ProcessingThread, self).__init__(name="Thread #" + str(thread_id))
        self.__thread_id = thread_id
        self.__data_collection_directory = data_collection_directory
        self.__file_name_prefix          = file_name_prefix
        self.__simulated_mask_directory  = simulated_mask_directory
        self.__energy                    = energy
        self.__source_distance           = source_distance
        self.__image_transfer_matrix     = image_transfer_matrix
        self.__verbose                   = verbose

    def run(self):
        def check_new_data(images_list):
            image_indexes      = []
            result_folder_list = glob.glob(os.path.join(os.path.dirname(images_list[0]), '*'))
            result_folder_list = [os.path.basename(f) for f in result_folder_list]

            for image in images_list:
                image_directory = os.path.basename(image).split('.tif')[0]
                if image_directory in result_folder_list: continue
                else: image_indexes.append(int(image_directory[-5:]))
            return image_indexes

        max_waiting_cycles = 60
        waiting_cycles     = 0

        while waiting_cycles < max_waiting_cycles:
            images_list   = glob.glob(os.path.join(self.__data_collection_directory, self.__file_name_prefix + '*.tif'), recursive=False)
            if len(images_list) == 0:
                waiting_cycles += 1
                print('Thread #' + str(self.__thread_id) + ' waiting for 1s for new data....')
            else:
                image_indexes = check_new_data(images_list)

                if len(image_indexes) == 0:
                    waiting_cycles += 1
                    print('Thread #' + str(self.__thread_id) + ' waiting for 1s for new data....')
                else:
                    random.shuffle(image_indexes)
                    if len(image_indexes) < 5: n = 1
                    else:                      n = 5

                    for image_index in image_indexes[0:n]: _process_image(self.__data_collection_directory,
                                                                          self.__file_name_prefix,
                                                                          self.__simulated_mask_directory,
                                                                          self.__energy,
                                                                          self.__source_distance,
                                                                          self.__image_transfer_matrix,
                                                                          image_index,
                                                                          self.__verbose)
            time.sleep(1)

        print('Thread #' + str(self.__thread_id) + ' completed')


def _get_image_data(data_collection_directory, file_name_prefix, mask_directory, energy, source_distance, image_transfer_matrix, image_index, verbose):
    dark = None
    flat = None
    image_path       = os.path.join(data_collection_directory, file_name_prefix + "%05i.tif" % image_index)
    mask_directory   = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory
    result_directory = os.path.join(os.path.dirname(image_path), os.path.basename(image_path).split('.tif')[0])

    # pattern simulation parameters
    pattern_path          = os.path.join(SCRIPT_DIRECTORY, 'mask', 'RanMask5umB0.npy')
    propagated_pattern    = os.path.join(mask_directory, 'propagated_pattern.npz')
    propagated_patternDet = os.path.join(mask_directory, 'propagated_patternDet.npz')

    crop                 = ' '.join([str(k) for k in [-1]])
    img_transfer_matrix  = ' '.join([str(k) for k in image_transfer_matrix])
    find_transfer_matrix = False
    p_x                  = PIXEL_SIZE
    det_array            = str(IMAGE_SIZE_PIXEL_HxV[1]) + " " + str(IMAGE_SIZE_PIXEL_HxV[0])
    pattern_size         = 4.942e-6  # 4.952e-6
    pattern_thickness    = 1.5e-6
    pattern_T            = 0.613
    d_prop               = 500e-3
    source_h             = 277e-6 / (60 / 1.5)
    source_v             = 10e-6 / (60 / 2)
    d_source_h           = source_distance[0]
    d_source_v           = source_distance[1]
    show_alignFigure     = False

    # reconstruction parameter initialization
    mode            = 'centralLine'  # area or centralLine
    lineWidth       = 10
    down_sampling   = 0.5
    method          = 'WXST'
    use_gpu         = True
    use_wavelet     = True
    wavelet_cut     = 1
    pyramid_level   = 1
    template_size   = 21
    window_search   = 20
    crop_boundary   = -1
    n_cores         = 16
    n_group         = 1
    verbose         = 1 if verbose else 0 # NO
    simple_analysis = 1

    # alignment or not, if '', no alignment, '--alignment' with alignment
    params = ['--GPU ' if use_gpu else ''] + ['--use_wavelet ' if use_wavelet else ''] + [
        '--show_alignFigure ' if show_alignFigure else ''] + ['--find_transferMatrix ' if find_transfer_matrix else '']
    params = ''.join([str(item) for item in params])

    command = 'python ' + os.path.join(SCRIPT_DIRECTORY, 'main.py') + \
              ' --img {} --dark {} --flat {} --result_folder {} --pattern_path {} ' \
              '--propagated_pattern {} --propagated_patternDet {} --crop {} --det_size {} ' \
              '--img_transfer_matrix {} --p_x {} --energy {} --pattern_size {} --pattern_thickness {} ' \
              '--pattern_T {} --d_source_v {} --d_source_h {} --source_v {} --source_h {} --d_prop {} ' \
              '--mode {} --lineWidth {} --down_sampling {} --method {} --wavelet_lv_cut {} ' \
              '--pyramid_level {} --template_size {} --window_searching {} ' \
              '--nCores {} --nGroup {} --verbose {} --simple_analysis {} --crop_boundary {} {} '.format(image_path, dark, flat, result_directory,
                                                                      pattern_path, propagated_pattern,
                                                                      propagated_patternDet, crop, det_array,
                                                                      img_transfer_matrix, p_x, energy,
                                                                      pattern_size, pattern_thickness, pattern_T,
                                                                      d_source_v, d_source_h,
                                                                      source_v, source_h, d_prop, mode, lineWidth,
                                                                      down_sampling, method,
                                                                      wavelet_cut, pyramid_level, template_size,
                                                                      window_search, n_cores,
                                                                      n_group, verbose, simple_analysis, crop_boundary, params)
    os.system(command)

    with open(os.path.join(result_directory, "raw_image.npy"), 'rb') as f: image = numpy.load(f, allow_pickle=False).T

    h_coord = numpy.linspace(-IMAGE_SIZE_PIXEL_HxV[0] / 2, IMAGE_SIZE_PIXEL_HxV[0] / 2, IMAGE_SIZE_PIXEL_HxV[0]) * PIXEL_SIZE * 1e3
    v_coord = numpy.linspace(-IMAGE_SIZE_PIXEL_HxV[1] / 2, IMAGE_SIZE_PIXEL_HxV[1] / 2, IMAGE_SIZE_PIXEL_HxV[1]) * PIXEL_SIZE * 1e3

    return image, h_coord, v_coord

def _process_image(data_collection_directory, file_name_prefix, mask_directory, energy, source_distance, image_transfer_matrix, image_index, verbose):
    dark = None
    flat = None
    image_path       = os.path.join(data_collection_directory, file_name_prefix + "%05i.tif" % image_index)
    mask_directory   = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory
    result_directory = os.path.join(os.path.dirname(image_path), os.path.basename(image_path).split('.tif')[0])

    # pattern simulation parameters
    pattern_path          = os.path.join(SCRIPT_DIRECTORY, 'mask' , 'RanMask5umB0.npy')
    propagated_pattern    = os.path.join(mask_directory, 'propagated_pattern.npz')
    propagated_patternDet = os.path.join(mask_directory, 'propagated_patternDet.npz')

    crop                 = ' '.join([str(k) for k in [-1]])
    img_transfer_matrix  = ' '.join([str(k) for k in image_transfer_matrix])
    find_transfer_matrix = False
    p_x                  = PIXEL_SIZE
    det_array            = str(IMAGE_SIZE_PIXEL_HxV[1]) + " " + str(IMAGE_SIZE_PIXEL_HxV[0])
    pattern_size         = 4.942e-6  # 4.952e-6
    pattern_thickness    = 1.5e-6
    pattern_T            = 0.613
    d_prop               = 500e-3
    source_h             = 277e-6 / (60 / 1.5)
    source_v             = 10e-6 / (60 / 2)
    d_source_h           = source_distance[0]
    d_source_v           = source_distance[1]
    show_alignFigure     = False

    # reconstruction parameter initialization
    mode            = 'centralLine'  # area or centralLine
    lineWidth       = 10
    down_sampling   = 0.5
    method          = 'WXST'
    use_gpu         = True
    use_wavelet     = True
    wavelet_cut     = 1
    pyramid_level   = 1
    template_size   = 21
    window_search   = 20
    crop_boundary   = -1
    n_cores         = 16
    n_group         = 1
    verbose         = 1 if verbose else 0 # NO
    simple_analysis = 0 # NO

    # alignment or not, if '', no alignment, '--alignment' with alignment
    params = ['--GPU ' if use_gpu else ''] + ['--use_wavelet ' if use_wavelet else ''] + [
        '--show_alignFigure ' if show_alignFigure else ''] + ['--find_transferMatrix ' if find_transfer_matrix else '']
    params = ''.join([str(item) for item in params])

    command = 'python ' + os.path.join(SCRIPT_DIRECTORY, 'main.py') + \
              ' --img {} --dark {} --flat {} --result_folder {} --pattern_path {} ' \
              '--propagated_pattern {} --propagated_patternDet {} --crop {} --det_size {} ' \
              '--img_transfer_matrix {} --p_x {} --energy {} --pattern_size {} --pattern_thickness {} ' \
              '--pattern_T {} --d_source_v {} --d_source_h {} --source_v {} --source_h {} --d_prop {} ' \
              '--mode {} --lineWidth {} --down_sampling {} --method {} --wavelet_lv_cut {} ' \
              '--pyramid_level {} --template_size {} --window_searching {} ' \
              '--nCores {} --nGroup {} --verbose {} --simple_analysis {} --crop_boundary {} {} '.format(image_path, dark, flat, result_directory,
                                                                      pattern_path, propagated_pattern,
                                                                      propagated_patternDet, crop, det_array,
                                                                      img_transfer_matrix, p_x, energy,
                                                                      pattern_size, pattern_thickness, pattern_T,
                                                                      d_source_v, d_source_h,
                                                                      source_v, source_h, d_prop, mode, lineWidth,
                                                                      down_sampling, method,
                                                                      wavelet_cut, pyramid_level, template_size,
                                                                      window_search, n_cores,
                                                                      n_group, verbose, simple_analysis, crop_boundary, params)
    os.system(command)

    print("Image " + file_name_prefix + "%05i.tif" % image_index + " processed")

def _generate_simulated_mask(data_collection_directory, file_name_prefix, mask_directory, energy, source_distance, image_index=1, verbose=False):
    dark = None
    flat = None
    image_path      = os.path.join(data_collection_directory, file_name_prefix + "%05i.tif" % image_index)
    mask_directory  = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory
    is_new_mask     = True

    if not os.path.exists(mask_directory): os.mkdir(mask_directory)

    if not os.path.exists(os.path.join(mask_directory, 'propagated_pattern.npz')) or \
       not os.path.exists(os.path.join(mask_directory, 'propagated_patternDet.npz')) or \
       not os.path.exists(os.path.join(mask_directory, "image_transfer_matrix.npy")):

        result_directory = os.path.join(os.path.dirname(image_path), os.path.basename(image_path).split('.tif')[0])

        # pattern simulation parameters
        pattern_path          = os.path.join(SCRIPT_DIRECTORY, 'mask', 'RanMask5umB0.npy')
        propagated_pattern    = None
        propagated_patternDet = None

        crop                 = ' '.join([str(k) for k in [-1]])
        find_transfer_matrix = True
        p_x                  = PIXEL_SIZE
        det_array            = str(IMAGE_SIZE_PIXEL_HxV[1]) + " " + str(IMAGE_SIZE_PIXEL_HxV[0])
        pattern_size         = 4.942e-6  # 4.952e-6
        pattern_thickness    = 1.5e-6
        pattern_T            = 0.613
        d_prop               = 500e-3
        source_h             = 277e-6 / (60 / 1.5)
        source_v             = 10e-6 / (60 / 2)
        d_source_h           = source_distance[0]
        d_source_v           = source_distance[1]
        show_alignFigure     = False

        # reconstruction parameter initialization
        mode            = 'centralLine'  # area or centralLine
        lineWidth       = 10
        down_sampling   = 0.5
        method          = 'WXST'
        use_gpu         = True
        use_wavelet     = True
        wavelet_cut     = 1
        pyramid_level   = 1
        template_size   = 21
        window_search   = 20
        crop_boundary   = -1
        n_cores         = 16
        n_group         = 1
        verbose         = 1 if verbose else 0 # NO
        simple_analysis = 0 # NO

        # alignment or not, if '', no alignment, '--alignment' with alignment
        params = ['--GPU ' if use_gpu else ''] + ['--use_wavelet ' if use_wavelet else ''] + [
            '--show_alignFigure ' if show_alignFigure else ''] + ['--find_transferMatrix ' if find_transfer_matrix else '']
        params = ''.join([str(item) for item in params])

        command = 'python ' + os.path.join(SCRIPT_DIRECTORY, 'main.py') + \
                  ' --img {} --dark {} --flat {} --result_folder {} --pattern_path {} ' \
                  '--propagated_pattern {} --propagated_patternDet {} --saving_path {} --crop {} --det_size {} ' \
                  '--p_x {} --energy {} --pattern_size {} --pattern_thickness {} ' \
                  '--pattern_T {} --d_source_v {} --d_source_h {} --source_v {} --source_h {} --d_prop {} ' \
                  '--d_source_recal --find_transferMatrix --mode {} --lineWidth {} --down_sampling {} --method {} --wavelet_lv_cut {} ' \
                  '--pyramid_level {} --template_size {} --window_searching {} ' \
                  '--nCores {} --nGroup {} --verbose {} --simple_analysis {} --crop_boundary {} {} '.format(image_path, dark, flat, result_directory,
                                                                          pattern_path, propagated_pattern, propagated_patternDet, mask_directory,
                                                                          crop, det_array, p_x, energy,
                                                                          pattern_size, pattern_thickness, pattern_T,
                                                                          d_source_v, d_source_h,
                                                                          source_v, source_h, d_prop, mode, lineWidth,
                                                                          down_sampling, method,
                                                                          wavelet_cut, pyramid_level, template_size,
                                                                          window_search, n_cores,
                                                                          n_group, verbose, simple_analysis, crop_boundary, params)
        os.system(command)

        print("Simulated mask generated in " + mask_directory)
    else:
        is_new_mask = False
        print("Simulated mask already generated in " + mask_directory)

    with open(os.path.join(mask_directory, "image_transfer_matrix.npy"), 'rb') as f: image_transfer_matrix = numpy.load(f, allow_pickle=False)

    return image_transfer_matrix.tolist(), is_new_mask
