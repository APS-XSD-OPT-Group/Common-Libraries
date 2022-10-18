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
import os, time
from aps.util.singleton import Singleton, synchronized_method

GREEN = "GREEN"
RED   = "RED"

class TrafficLightFacade():
    def set_red_light(self): raise NotImplementedError()
    def set_green_light(self): raise NotImplementedError()
    def is_green_light(self): raise NotImplementedError()
    def is_red_light(self): raise NotImplementedError()

class __TrafficLightFile(TrafficLightFacade):

    def __init__(self, application_name, **kwargs):
        try:    common_directory = kwargs["common_directory"]
        except: common_directory = os.curdir
        try:    self.__traffic_light_file = os.path.join(common_directory, kwargs["file_name"] + ".lock")
        except: self.__traffic_light_file = os.path.join(common_directory, application_name    + ".lock")
        try:    self.__max_number_of_waiting_cycles = kwargs["max_number_of_waiting_cycles"]
        except: self.__max_number_of_waiting_cycles = 10

        if not os.path.exists(self.__traffic_light_file): self.set_green_light()

    def __change_color(self, color : str):
        waiting_cycle = 0
        while waiting_cycle < self.__max_number_of_waiting_cycles:
            try:
                f = open(self.__traffic_light_file, 'w')
                f.write(color)
                f.close()
                return
            except IOError as e:
                if "already opened" in str(e):
                    time.sleep(0.5)
                    waiting_cycle += 1
                else: raise e

    def __get_color(self):
        waiting_cycle = 0
        while waiting_cycle < self.__max_number_of_waiting_cycles:
            try:
                f = open(self.__traffic_light_file, 'r')
                color = f.readline()
                f.close()

                return color
            except IOError as e:
                if "already opened" in str(e):
                    time.sleep(0.5)
                    waiting_cycle += 1
                else: raise e


    @synchronized_method
    def set_red_light(self): self.__change_color(RED)

    @synchronized_method
    def set_green_light(self):self.__change_color(GREEN)

    @synchronized_method
    def is_red_light(self): return self.__get_color() == RED

    @synchronized_method
    def is_green_light(self): return self.__get_color() == GREEN

from aps.util.registry import GenericRegistry

@Singleton
class __TrafficLightRegistry(GenericRegistry):
    def __init__(self):
        GenericRegistry.__init__(self, registry_name="Traffic-Light")

    @synchronized_method
    def register_traffic_light(self, traffic_light_facade_instance, application_name=None): super().register_instance(traffic_light_facade_instance, application_name, False)

    @synchronized_method
    def reset(self, application_name=None): super().reset(application_name)

    def get_traffic_light_instance(self, application_name=None): return super().get_instance(application_name)

# -----------------------------------------------------
# Factory Methods

def register_traffic_light_instance(reset=False, application_name=None, **kwargs):
    if reset: __TrafficLightRegistry.Instance().reset(application_name)
    __TrafficLightRegistry.Instance().register_traffic_light(__TrafficLightFile(application_name, **kwargs), application_name)

def get_registered_traffic_light_instance(application_name=None):
    return __TrafficLightRegistry.Instance().get_traffic_light_instance(application_name)
