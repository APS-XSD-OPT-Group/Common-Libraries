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

from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance

from aps.common.measurment.beamline.image_collector import get_default_file_name_prefix
from aps.common.measurment.beamline.wf import SCRIPT_DIRECTORY

APPLICATION_NAME = "IMAGE-PROCESSOR"

register_ini_instance(IniMode.LOCAL_FILE,
                      ini_file_name="image_processor.ini",
                      application_name=APPLICATION_NAME,
                      verbose=False)
ini_file = get_registered_ini_instance(APPLICATION_NAME)

PYTHON                = ini_file.get_string_from_ini( section="Python", key="Python-Executable", default="python")

PIXEL_SIZE            = ini_file.get_float_from_ini(  section="Detector", key="Pixel-Size", default=0.65e-6)
IMAGE_SIZE_PIXEL_HxV  = ini_file.get_list_from_ini(   section="Detector", key="Image-Size", default=[2160, 2560], type=int)

PATTERN_SIZE          = ini_file.get_float_from_ini(  section="Mask", key="Pattern-Size",         default=4.942e-6)
PATTERN_THICKNESS     = ini_file.get_float_from_ini(  section="Mask", key="Pattern-Thickness",    default=1.5e-6)
PATTERN_T             = ini_file.get_float_from_ini(  section="Mask", key="Pattern-Transmission", default=0.613)
RAN_MASK              = ini_file.get_string_from_ini( section="Mask", key="Pattern-Image",        default='RanMask5umB0.npy')
D_PROP                = ini_file.get_float_from_ini(  section="Mask", key="Propagation-Distance", default=500e-3)

SOURCE_V              = ini_file.get_float_from_ini(  section="Source", key="Source-Size-V",     default=6.925e-6)
SOURCE_H              = ini_file.get_float_from_ini(  section="Source", key="Source-Size-H",     default=0.333e-6)
SOURCE_DISTANCE_V     = ini_file.get_float_from_ini(  section="Source", key="Source-Distance-V", default=1.5)
SOURCE_DISTANCE_H     = ini_file.get_float_from_ini(  section="Source", key="Source-Distance-H", default=1.5)

D_SOURCE_RECAL        = ini_file.get_boolean_from_ini(section="Execution", key="Source-Distance-Recalculation", default=True)
CROP                  = ini_file.get_list_from_ini(   section="Execution", key="Crop",                          default=[-1], type=int)
ESTIMATION_METHOD     = ini_file.get_string_from_ini( section="Execution", key="Estimation-Method",             default='simple_speckle')

MODE                  = ini_file.get_string_from_ini( section="Reconstruction", key="Mode",           default='centralLine')
LINE_WIDTH            = ini_file.get_int_from_ini(    section="Reconstruction", key="Line-Width",     default=10)
LINE_DIRECTION        = ini_file.get_string_from_ini( section="Reconstruction", key="Line-Direction", default='b')
DOWN_SAMPLING         = ini_file.get_float_from_ini(  section="Reconstruction", key="Down-Sampling",  default=0.5)
METHOD                = ini_file.get_string_from_ini( section="Reconstruction", key="Method",         default='WXST')
USE_GPU               = ini_file.get_boolean_from_ini(section="Reconstruction", key="Use-Gpu",        default=True)
USE_WAVELET           = ini_file.get_boolean_from_ini(section="Reconstruction", key="Use-Wavelet",    default=True)
WAVELET_CUT           = ini_file.get_int_from_ini(    section="Reconstruction", key="Wavelet-Cut",    default=1)
PYRAMID_LEVEL         = ini_file.get_int_from_ini(    section="Reconstruction", key="Pyramid-Level",  default=1)
TEMPLATE_SIZE         = ini_file.get_int_from_ini(    section="Reconstruction", key="Template-Size",  default=21)
WINDOW_SEARCH         = ini_file.get_int_from_ini(    section="Reconstruction", key="Window-Search",  default=20)
CROP_BOUNDARY         = ini_file.get_int_from_ini(    section="Reconstruction", key="Crop-Boundary",  default=-1)
N_CORES               = ini_file.get_int_from_ini(    section="Reconstruction", key="N-Cores",        default=16)
N_GROUP               = ini_file.get_int_from_ini(    section="Reconstruction", key="N-Group",        default=1)

IMAGE_TRANSFER_MATRIX = ini_file.get_list_from_ini(   section="Output", key="Image-Transfer-Matrix", default=[0, 1, 0], type=int)
SHOW_ALIGN_FIGURE     = ini_file.get_boolean_from_ini(section="Output", key="Show-Align-Figure",     default=False)

ini_file.set_value_at_ini(section="Python",   key="Python-Executable", value=PYTHON)

ini_file.set_value_at_ini(section="Detector", key="Pixel-Size", value=PIXEL_SIZE)
ini_file.set_list_at_ini( section="Detector", key="Image-Size", values_list=IMAGE_SIZE_PIXEL_HxV)

ini_file.set_value_at_ini(section="Mask", key="Pattern-Size",         value=PATTERN_SIZE)
ini_file.set_value_at_ini(section="Mask", key="Pattern-Thickness",    value=PATTERN_THICKNESS)
ini_file.set_value_at_ini(section="Mask", key="Pattern-Transmission", value=PATTERN_T)
ini_file.set_value_at_ini(section="Mask", key="Pattern-Image",        value=RAN_MASK)
ini_file.set_value_at_ini(section="Mask", key="Propagation-Distance", value=D_PROP)

ini_file.set_value_at_ini(section="Source", key="Source-Size-V",        value=SOURCE_V)
ini_file.set_value_at_ini(section="Source", key="Source-Size-H",        value=SOURCE_H)
ini_file.set_value_at_ini(section="Source", key="Source-Distance-V",    value=SOURCE_DISTANCE_V)
ini_file.set_value_at_ini(section="Source", key="Source-Distance-H",    value=SOURCE_DISTANCE_H)

ini_file.set_value_at_ini(section="Execution", key="Source-Distance-Recalculation", value=D_SOURCE_RECAL)
ini_file.set_list_at_ini( section="Execution", key="Crop",                          values_list=CROP)
ini_file.set_value_at_ini(section="Execution", key="Estimation-Method",             value=ESTIMATION_METHOD)

ini_file.set_value_at_ini(section="Reconstruction", key="Mode",           value=MODE)
ini_file.set_value_at_ini(section="Reconstruction", key="Line-Width",     value=LINE_WIDTH   )
ini_file.set_value_at_ini(section="Reconstruction", key="Line-Direction", value=LINE_DIRECTION)
ini_file.set_value_at_ini(section="Reconstruction", key="Down-Sampling",  value=DOWN_SAMPLING)
ini_file.set_value_at_ini(section="Reconstruction", key="Method",         value=METHOD       )
ini_file.set_value_at_ini(section="Reconstruction", key="Use-Gpu",        value=USE_GPU      )
ini_file.set_value_at_ini(section="Reconstruction", key="Use-Wavelet",    value=USE_WAVELET  )
ini_file.set_value_at_ini(section="Reconstruction", key="Wavelet-Cut",    value=WAVELET_CUT  )
ini_file.set_value_at_ini(section="Reconstruction", key="Pyramid-Level",  value=PYRAMID_LEVEL)
ini_file.set_value_at_ini(section="Reconstruction", key="Template-Size",  value=TEMPLATE_SIZE)
ini_file.set_value_at_ini(section="Reconstruction", key="Window-Search",  value=WINDOW_SEARCH)
ini_file.set_value_at_ini(section="Reconstruction", key="Crop-Boundary",  value=CROP_BOUNDARY)
ini_file.set_value_at_ini(section="Reconstruction", key="N-Cores",        value=N_CORES      )
ini_file.set_value_at_ini(section="Reconstruction", key="N-Group",        value=N_GROUP      )

ini_file.set_list_at_ini( section="Output", key="Image-Transfer-Matrix", values_list=IMAGE_TRANSFER_MATRIX)
ini_file.set_value_at_ini(section="Output", key="Show-Align-Figure",     value=SHOW_ALIGN_FIGURE)

ini_file.push()

class ImageProcessor():
    def __init__(self,
                 data_collection_directory,
                 file_name_prefix=get_default_file_name_prefix(),
                 simulated_mask_directory=None,
                 energy=20000.0):
        self.__data_collection_directory = data_collection_directory
        self.__file_name_prefix          = file_name_prefix
        self.__simulated_mask_directory  = simulated_mask_directory
        self.__energy                    = energy
        self.__source_distance           = [SOURCE_DISTANCE_H, SOURCE_DISTANCE_V]
        self.__image_transfer_matrix     = IMAGE_TRANSFER_MATRIX

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
            images_list   = glob.glob(os.path.join(self.__data_collection_directory, self.__file_name_prefix + '_*.tif'), recursive=False)
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
    image_path       = os.path.join(data_collection_directory, file_name_prefix + "_%05i.tif" % image_index)
    mask_directory   = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory
    result_directory = os.path.join(os.path.dirname(image_path), os.path.basename(image_path).split('.tif')[0])

    # pattern simulation parameters
    pattern_path          = os.path.join(SCRIPT_DIRECTORY, 'mask', RAN_MASK)
    propagated_pattern    = os.path.join(mask_directory, 'propagated_pattern.npz')
    propagated_patternDet = os.path.join(mask_directory, 'propagated_patternDet.npz')

    crop                 = ' '.join([str(k) for k in CROP])
    img_transfer_matrix  = ' '.join([str(k) for k in image_transfer_matrix])
    find_transfer_matrix = False
    p_x                  = PIXEL_SIZE
    det_array            = str(IMAGE_SIZE_PIXEL_HxV[1]) + " " + str(IMAGE_SIZE_PIXEL_HxV[0])
    pattern_size         = PATTERN_SIZE
    pattern_thickness    = PATTERN_THICKNESS
    pattern_T            = PATTERN_T
    d_prop               = D_PROP
    source_h             = SOURCE_H
    source_v             = SOURCE_V
    d_source_h           = source_distance[0]
    d_source_v           = source_distance[1]
    show_align_figure    = SHOW_ALIGN_FIGURE

    # reconstruction parameter initialization
    mode            = MODE  # area or centralLine
    lineWidth       = LINE_WIDTH
    lineDirection   = LINE_DIRECTION
    down_sampling   = DOWN_SAMPLING
    method          = METHOD
    use_gpu         = USE_GPU
    use_wavelet     = USE_WAVELET
    wavelet_cut     = WAVELET_CUT
    pyramid_level   = PYRAMID_LEVEL
    template_size   = TEMPLATE_SIZE
    window_search   = WINDOW_SEARCH
    crop_boundary   = CROP_BOUNDARY
    n_cores         = N_CORES
    n_group         = N_GROUP
    verbose         = 0 if verbose else 1 # NO
    simple_analysis = 1

    # alignment or not, if '', no alignment, '--alignment' with alignment
    params = ['--GPU ' if use_gpu else ''] + ['--use_wavelet ' if use_wavelet else ''] + [
        '--show_alignFigure ' if show_align_figure else ''] + ['--find_transferMatrix ' if find_transfer_matrix else '']
    params = ''.join([str(item) for item in params])

    command = PYTHON + ' '  + os.path.join(SCRIPT_DIRECTORY, 'main.py') + \
              ' --img {} --dark {} --flat {} --result_folder {} --pattern_path {} ' \
              '--propagated_pattern {} --propagated_patternDet {} --crop {} --det_size {} ' \
              '--img_transfer_matrix {} --p_x {} --energy {} --pattern_size {} --pattern_thickness {} ' \
              '--pattern_T {} --d_source_v {} --d_source_h {} --source_v {} --source_h {} --d_prop {} ' \
              '--mode {} --lineWidth {} --lineDirection {} --down_sampling {} --method {} --wavelet_lv_cut {} ' \
              '--pyramid_level {} --template_size {} --window_searching {} ' \
              '--nCores {} --nGroup {} --verbose {} --simple_analysis {} --crop_boundary {} {} '.format(image_path, dark, flat, result_directory,
                                                                      pattern_path, propagated_pattern,
                                                                      propagated_patternDet, crop, det_array,
                                                                      img_transfer_matrix, p_x, energy,
                                                                      pattern_size, pattern_thickness, pattern_T,
                                                                      d_source_v, d_source_h,
                                                                      source_v, source_h, d_prop, mode, lineWidth, lineDirection,
                                                                      down_sampling, method,
                                                                      wavelet_cut, pyramid_level, template_size,
                                                                      window_search, n_cores,
                                                                      n_group, verbose, simple_analysis, crop_boundary, params)
    ret_val = os.system(command)

    if ret_val != 0: raise Exception("Wavefront analysis failed")

    with open(os.path.join(result_directory, "raw_image.npy"), 'rb') as f: image = numpy.load(f, allow_pickle=False).T

    h_coord = numpy.linspace(-IMAGE_SIZE_PIXEL_HxV[0] / 2, IMAGE_SIZE_PIXEL_HxV[0] / 2, IMAGE_SIZE_PIXEL_HxV[0]) * PIXEL_SIZE * 1e3
    v_coord = numpy.linspace(-IMAGE_SIZE_PIXEL_HxV[1] / 2, IMAGE_SIZE_PIXEL_HxV[1] / 2, IMAGE_SIZE_PIXEL_HxV[1]) * PIXEL_SIZE * 1e3

    return image, h_coord, v_coord

def _process_image(data_collection_directory, file_name_prefix, mask_directory, energy, source_distance, image_transfer_matrix, image_index, verbose):
    dark = None
    flat = None
    image_path       = os.path.join(data_collection_directory, file_name_prefix + "_%05i.tif" % image_index)
    mask_directory   = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory
    result_directory = os.path.join(os.path.dirname(image_path), os.path.basename(image_path).split('.tif')[0])

    # pattern simulation parameters
    pattern_path          = os.path.join(SCRIPT_DIRECTORY, 'mask' , RAN_MASK)
    propagated_pattern    = os.path.join(mask_directory, 'propagated_pattern.npz')
    propagated_patternDet = os.path.join(mask_directory, 'propagated_patternDet.npz')

    crop                 = ' '.join([str(k) for k in CROP])
    img_transfer_matrix  = ' '.join([str(k) for k in image_transfer_matrix])
    find_transfer_matrix = False
    p_x                  = PIXEL_SIZE
    det_array            = str(IMAGE_SIZE_PIXEL_HxV[1]) + " " + str(IMAGE_SIZE_PIXEL_HxV[0])
    pattern_size         = PATTERN_SIZE
    pattern_thickness    = PATTERN_THICKNESS
    pattern_T            = PATTERN_T
    d_prop               = D_PROP
    source_h             = SOURCE_H
    source_v             = SOURCE_V
    d_source_h           = source_distance[0]
    d_source_v           = source_distance[1]
    show_align_figure    = SHOW_ALIGN_FIGURE

    # reconstruction parameter initialization
    mode            = MODE  # area or centralLine
    lineWidth       = LINE_WIDTH
    lineDirection   = LINE_DIRECTION
    down_sampling   = DOWN_SAMPLING
    method          = METHOD
    use_gpu         = USE_GPU
    use_wavelet     = USE_WAVELET
    wavelet_cut     = WAVELET_CUT
    pyramid_level   = PYRAMID_LEVEL
    template_size   = TEMPLATE_SIZE
    window_search   = WINDOW_SEARCH
    crop_boundary   = CROP_BOUNDARY
    n_cores         = N_CORES
    n_group         = N_GROUP
    verbose         = 0 if verbose else 1 # NO
    simple_analysis = 0 # NO

    # alignment or not, if '', no alignment, '--alignment' with alignment
    params = ['--GPU ' if use_gpu else ''] + ['--use_wavelet ' if use_wavelet else ''] + [
        '--show_alignFigure ' if show_align_figure else ''] + ['--find_transferMatrix ' if find_transfer_matrix else '']
    params = ''.join([str(item) for item in params])

    command = PYTHON + ' '  + os.path.join(SCRIPT_DIRECTORY, 'main.py') + \
              ' --img {} --dark {} --flat {} --result_folder {} --pattern_path {} ' \
              '--propagated_pattern {} --propagated_patternDet {} --crop {} --det_size {} ' \
              '--img_transfer_matrix {} --p_x {} --energy {} --pattern_size {} --pattern_thickness {} ' \
              '--pattern_T {} --d_source_v {} --d_source_h {} --source_v {} --source_h {} --d_prop {} ' \
              '--mode {} --lineWidth {} --lineDirection {} --down_sampling {} --method {} --wavelet_lv_cut {} ' \
              '--pyramid_level {} --template_size {} --window_searching {} ' \
              '--nCores {} --nGroup {} --verbose {} --simple_analysis {} --crop_boundary {} {} '.format(image_path, dark, flat, result_directory,
                                                                      pattern_path, propagated_pattern,
                                                                      propagated_patternDet, crop, det_array,
                                                                      img_transfer_matrix, p_x, energy,
                                                                      pattern_size, pattern_thickness, pattern_T,
                                                                      d_source_v, d_source_h,
                                                                      source_v, source_h, d_prop, mode, lineWidth, lineDirection,
                                                                      down_sampling, method,
                                                                      wavelet_cut, pyramid_level, template_size,
                                                                      window_search, n_cores,
                                                                      n_group, verbose, simple_analysis, crop_boundary, params)
    ret_val = os.system(command)

    if ret_val != 0: raise Exception("Wavefront analysis failed")
    else:            print("Image " + file_name_prefix + "_%05i.tif" % image_index + " processed")

def _generate_simulated_mask(data_collection_directory, file_name_prefix, mask_directory, energy, source_distance, image_index=1, verbose=False):
    dark = None
    flat = None
    image_path      = os.path.join(data_collection_directory, file_name_prefix + "_%05i.tif" % image_index)
    mask_directory  = os.path.join(data_collection_directory, "simulated_mask") if mask_directory is None else mask_directory
    is_new_mask     = True

    if not os.path.exists(mask_directory): os.mkdir(mask_directory)

    if not os.path.exists(os.path.join(mask_directory, 'propagated_pattern.npz')) or \
       not os.path.exists(os.path.join(mask_directory, 'propagated_patternDet.npz')) or \
       not os.path.exists(os.path.join(mask_directory, "image_transfer_matrix.npy")):

        result_directory = os.path.join(os.path.dirname(image_path), os.path.basename(image_path).split('.tif')[0])

        # pattern simulation parameters
        pattern_path          = os.path.join(SCRIPT_DIRECTORY, 'mask', RAN_MASK)
        propagated_pattern    = None
        propagated_patternDet = None
        estimation_method     = ESTIMATION_METHOD

        crop                 = ' '.join([str(k) for k in CROP])
        find_transfer_matrix = True
        p_x                  = PIXEL_SIZE
        det_array            = str(IMAGE_SIZE_PIXEL_HxV[1]) + " " + str(IMAGE_SIZE_PIXEL_HxV[0])
        pattern_size         = PATTERN_SIZE
        pattern_thickness    = PATTERN_THICKNESS
        pattern_T            = PATTERN_T
        d_prop               = D_PROP
        source_h             = SOURCE_H
        source_v             = SOURCE_V
        d_source_h           = source_distance[0]
        d_source_v           = source_distance[1]
        show_align_figure    = SHOW_ALIGN_FIGURE

        # reconstruction parameter initialization
        mode = MODE  # area or centralLine
        lineWidth = LINE_WIDTH
        lineDirection = LINE_DIRECTION
        down_sampling = DOWN_SAMPLING
        method = METHOD
        use_gpu = USE_GPU
        use_wavelet = USE_WAVELET
        wavelet_cut = WAVELET_CUT
        pyramid_level = PYRAMID_LEVEL
        template_size = TEMPLATE_SIZE
        window_search = WINDOW_SEARCH
        crop_boundary = CROP_BOUNDARY
        n_cores = N_CORES
        n_group = N_GROUP
        verbose         = 0 if verbose else 1 # NO
        simple_analysis = 0 # NO

        # alignment or not, if '', no alignment, '--alignment' with alignment
        params = ['--GPU ' if use_gpu else ''] + ['--use_wavelet ' if use_wavelet else ''] + \
                 ['--show_alignFigure ' if show_align_figure else ''] + \
                 ['--find_transferMatrix ' if find_transfer_matrix else ''] + \
                 ['--d_source_recal ' if D_SOURCE_RECAL else '']
        params = ''.join([str(item) for item in params])

        command = PYTHON + ' '  + os.path.join(SCRIPT_DIRECTORY, 'main.py') + \
                  ' --img {} --dark {} --flat {} --result_folder {} --pattern_path {} ' \
                  '--propagated_pattern {} --propagated_patternDet {} --estimation_method {} --saving_path {} --crop {} --det_size {} ' \
                  '--p_x {} --energy {} --pattern_size {} --pattern_thickness {} ' \
                  '--pattern_T {} --d_source_v {} --d_source_h {} --source_v {} --source_h {} --d_prop {} ' \
                  '--mode {} --lineWidth {} --lineDirection {} --down_sampling {} --method {} --wavelet_lv_cut {} ' \
                  '--pyramid_level {} --template_size {} --window_searching {} ' \
                  '--nCores {} --nGroup {} --verbose {} --simple_analysis {} --crop_boundary {} {} '.format(image_path, dark, flat, result_directory,
                                                                          pattern_path, propagated_pattern, propagated_patternDet, estimation_method, mask_directory,
                                                                          crop, det_array, p_x, energy,
                                                                          pattern_size, pattern_thickness, pattern_T,
                                                                          d_source_v, d_source_h,
                                                                          source_v, source_h, d_prop, mode, lineWidth, lineDirection,
                                                                          down_sampling, method,
                                                                          wavelet_cut, pyramid_level, template_size,
                                                                          window_search, n_cores,
                                                                          n_group, verbose, simple_analysis, crop_boundary, params)
        ret_val = os.system(command)

        if ret_val != 0: raise Exception("Wavefront analysis failed")
        else:            print("Simulated mask generated in " + mask_directory)
    else:
        is_new_mask = False
        print("Simulated mask already generated in " + mask_directory)

    with open(os.path.join(mask_directory, "image_transfer_matrix.npy"), 'rb') as f: image_transfer_matrix = numpy.load(f, allow_pickle=False)

    return image_transfer_matrix.tolist(), is_new_mask
