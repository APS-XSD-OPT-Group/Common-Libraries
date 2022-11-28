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
import time
from epics import PV
import pickle
from collections import OrderedDict

from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance

APPLICATION_NAME = "IMAGE-COLLECTOR"

register_ini_instance(IniMode.LOCAL_FILE,
                      ini_file_name="image_collector.ini",
                      application_name=APPLICATION_NAME,
                      verbose=False)
ini_file = get_registered_ini_instance(APPLICATION_NAME)

WAIT_TIME         = ini_file.get_float_from_ini(section="Execution", key="Wait-Time",     default=0.1)
EXPOSURE_TIME     = ini_file.get_float_from_ini(section="Execution", key="Exposure-Time", default=0.3)
FILE_NAME_PREFIX  = "sample_" + str(int(EXPOSURE_TIME*1000)) + "ms_"

ini_file.set_value_at_ini(section="Execution",   key="Wait-Time",     value=WAIT_TIME)
ini_file.set_value_at_ini(section="Execution",   key="Exposure-Time", value=EXPOSURE_TIME)
ini_file.push()

IMAGE_COLLECTOR_STATUS_FILE = "image_collector_status.pkl"

class ImageCollector():

    def __init__(self, measurement_directory, exposure_time=EXPOSURE_TIME, detector_delay=None, mocking_mode=False):
        self.__exposure_time         = exposure_time
        self.__measurement_directory = measurement_directory
        self.__mocking_mode          = mocking_mode

        if not self.__mocking_mode:
            self.__PV_dict = dict(
                andor_cam_acquire        = PV("dp_andor3_skylark:cam1:Acquire"),  # 0="Done", 1="Acquire"
                andor_cam_exposure_time  = PV("dp_andor3_skylark:cam1:AcquireTime"),
                andor_cam_image_mode     = PV("dp_andor3_skylark:cam1:ImageMode"),    # "Fixed" or "Continuous"
                andor_tiff_filename      = PV("dp_andor3_skylark:TIFF1:FileName"),
                andor_tiff_filepath      = PV("dp_andor3_skylark:TIFF1:FilePath"),
                andor_tiff_filenumber    = PV("dp_andor3_skylark:TIFF1:FileNumber"),
                andor_tiff_autosave      = PV("dp_andor3_skylark:TIFF1:AutoSave"),
                andor_tiff_savefile      = PV("dp_andor3_skylark:TIFF1:WriteFile"),
                andor_tiff_autoincrement = PV("dp_andor3_skylark:TIFF1:AutoIncrement")
            )

            if detector_delay is None:
                self.__has_delay = False
            else:
                self.__has_delay      = True
                self.__detector_delay = detector_delay

            self.__detector_stop()
            self.__set_andor_defaults(1)
        else:
            print("ImageCollector initialized in Mocking Mode")

    def __to_pickle_file(self):
        if not self.__mocking_mode:
            self.__detector_stop()

            dictionary = OrderedDict()
            dictionary["andor_cam_image_mode"]     = self.__PV_dict["andor_cam_image_mode"].get()
            dictionary["andor_cam_exposure_time"]  = self.__PV_dict["andor_cam_exposure_time"].get()
            dictionary["andor_tiff_filepath"]      = self.__PV_dict["andor_tiff_filepath"].get()
            dictionary["andor_tiff_filenumber"]    = self.__PV_dict["andor_tiff_filenumber"].get()
            dictionary["andor_tiff_savefile"]      = self.__PV_dict["andor_tiff_savefile"].get()
            dictionary["andor_tiff_autoincrement"] = self.__PV_dict["andor_tiff_autoincrement"].get()

            file = open(IMAGE_COLLECTOR_STATUS_FILE, 'wb')
            pickle.dump(dictionary, file)
            file.close()

    def __from_pickle_file(self):
        if not self.__mocking_mode:
            self.__detector_stop()

            file = open(IMAGE_COLLECTOR_STATUS_FILE, 'rb')
            dictionary = pickle.load(file)
            file.close()

            self.__PV_dict["andor_cam_image_mode"].put(    dictionary["andor_cam_image_mode"])
            self.__PV_dict["andor_cam_exposure_time"].put( dictionary["andor_cam_exposure_time"])
            self.__PV_dict["andor_tiff_filepath"].put(     dictionary["andor_tiff_filepath"])
            self.__PV_dict["andor_tiff_filenumber"].put(   dictionary["andor_tiff_filenumber"])
            self.__PV_dict["andor_tiff_savefile"].put(     dictionary["andor_tiff_savefile"])
            self.__PV_dict["andor_tiff_autoincrement"].put(dictionary["andor_tiff_autoincrement"])

    def save_status(self):
        self.__to_pickle_file()

    def restore_status(self):
        self.__detector_stop()  # 1 waiting time
        self.__from_pickle_file()

    def collect_single_shot_image(self, index=1):
        if not self.__mocking_mode:
            self.__initialize_current_image(index)

            self.__detector_acquire() # 2 waiting time + exposure time
        else:
            time.sleep(self.get_total_acquisition_time())
            print("Mocking Mode: collected image #" + str(index))

    def end_collection(self): # to be done at the end of the data collection
        if not self.__mocking_mode:
            self.__PV_dict["andor_tiff_autosave"].put("No")
            self.__PV_dict["andor_tiff_autoincrement"].put("Yes")

    def get_total_acquisition_time(self):
        return 3*WAIT_TIME + self.__exposure_time

    def __detector_acquire(self):
        self.__detector_start()
        self.__detector_delay()

    def __detector_start(self):
        self.__PV_dict["andor_cam_acquire"].put(1)
        time.sleep(WAIT_TIME)

    def __detector_stop(self):
        self.__PV_dict["andor_cam_acquire"].put(0)
        time.sleep(WAIT_TIME)

    def __detector_acquiring(self):
        return self.__PV_dict["andor_cam_acquire"].get() in (1, "Acquiring")

    def __detector_done(self):
        return self.__PV_dict["andor_cam_acquire"].get() in (0, "Done")

    def __initialize_current_image(self, index):
        self.__detector_stop()  # 1 waiting time
        self.__set_andor_defaults(index)

    def __set_andor_defaults(self, index):
        self.__PV_dict["andor_cam_image_mode"].put("Fixed")
        self.__PV_dict["andor_cam_exposure_time"].put(self.__exposure_time)
        self.__PV_dict["andor_tiff_filepath"].put(self.__measurement_directory)
        self.__PV_dict["andor_tiff_autosave"].put("Yes")
        self.__PV_dict["andor_tiff_autoincrement"].put("No")
        self.__PV_dict["andor_tiff_filename"].put('sample_' + str(int(self.__exposure_time * 1000)) + 'ms')
        if index > 0: self.__PV_dict["andor_tiff_filenumber"].put(index)

    def __detector_delay(self):
        if not self.__has_delay:
            time.sleep(self.__exposure_time + WAIT_TIME)
        else:
            time.sleep(WAIT_TIME)  # wait for the detector to start
            kk = 0
            while self.__detector_acquiring():
                time.sleep(self.__detector_delay)
                kk += self.__detector_delay
                if kk > 120:
                    self.__detector_stop()
                    time.sleep(1)
                    self.__detector_start()
                    time.sleep(1)
                    kk = 0
