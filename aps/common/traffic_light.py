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
import json
from collections import OrderedDict

from aps.common.singleton import Singleton, synchronized_method

GREEN = "GREEN"
RED   = "RED"

class TrafficLightException(Exception):
    def __init__(self, message=None): super(TrafficLightException, self).__init__(message)


class TrafficLightFacade():
    def set_status_running(self): raise NotImplementedError()
    def release_status_running(self): raise NotImplementedError()
    def is_status_running(self): raise NotImplementedError()
    def request_red_light(self): raise NotImplementedError()
    def set_green_light(self): raise NotImplementedError()
    def is_green_light(self): raise NotImplementedError()

class __TrafficLightFile(TrafficLightFacade):

    def __init__(self, application_name, **kwargs):
        try:    common_directory = kwargs["common_directory"]
        except: common_directory = os.curdir
        try:    self.__traffic_light_file = os.path.join(common_directory, kwargs["file_name"] + ".lock")
        except: self.__traffic_light_file = os.path.join(common_directory, application_name    + ".lock")
        try:    self.__file_access_attempts = kwargs["file_access_attempts"]
        except: self.__file_access_attempts = 10
        try:    self.__max_wait_cycles = kwargs["max_wait_time"]*600
        except: self.__max_wait_cycles = 120*600 # 120 minutes

        self.__internal_dictionary = OrderedDict()

        if not os.path.exists(self.__traffic_light_file):
            self.__internal_dictionary["is_status_running"]   = False
            self.__internal_dictionary["traffic_light_color"] = GREEN
            self.__to_json_file()
        else:
            self.__from_json_file()

    def __to_json_file(self):
        json_content = json.dumps(self.__internal_dictionary, indent=4, separators=(',', ': '))
        waiting_cycle = 0
        while waiting_cycle < self.__file_access_attempts:
            try:
                f = open(self.__traffic_light_file, 'w')
                f.write(json_content)
                f.close()
                return
            except IOError as e:
                if "already opened" in str(e):
                    time.sleep(0.5)
                    waiting_cycle += 1
                else:
                    raise e

    def __from_json_file(self):
        waiting_cycle = 0
        while waiting_cycle < self.__file_access_attempts:
            try:
                f = open(self.__traffic_light_file, 'r')
                text = f.read()
                f.close()
                json_content = json.loads(text)

                self.__internal_dictionary["is_status_running"]   = json_content["is_status_running"]
                self.__internal_dictionary["traffic_light_color"] = json_content["traffic_light_color"]

                return
            except IOError as e:
                if "already opened" in str(e):
                    time.sleep(0.5)
                    waiting_cycle += 1
                else:
                    raise e

    def __change_color(self, color):
        self.__internal_dictionary["traffic_light_color"] = color
        self.__to_json_file()

    def __change_status_running(self, running):
        self.__internal_dictionary["is_status_running"] = running
        self.__to_json_file()

    def __get_color(self):
        self.__from_json_file()
        return self.__internal_dictionary["traffic_light_color"]

    def __get_is_status_running(self):
        self.__from_json_file()
        return self.__internal_dictionary["is_status_running"]

    @synchronized_method
    def set_status_running(self):
        waiting_cycle = 0
        while waiting_cycle < self.__max_wait_cycles:
            if self.is_green_light():
                self.__change_status_running(True)
                print("Status set to Running")
                return
            else:
                if waiting_cycle % 600 == 0: print("Red Light: waiting 60 seconds")
                time.sleep(0.1)
                waiting_cycle += 1

        raise TrafficLightException("Green light was never given during the " + str(self.__max_wait_cycles) + " 1 minute waiting cycles")

    @synchronized_method
    def release_status_running(self):
        if self.is_status_running():
            self.__change_status_running(False)
            time.sleep(0.5)
            print("Status set to Not Running")
        else:
            print("Status is already Not Running")

    @synchronized_method
    def is_status_running(self):
        return self.__get_is_status_running() == True

    @synchronized_method
    def request_red_light(self):
        waiting_cycle = 0
        while waiting_cycle < self.__max_wait_cycles:
            if not self.is_status_running():
                self.__change_color(RED)
                print("Light set to Red")
                return
            else:
                if waiting_cycle % 600 == 0: print("Red Light: waiting 60 seconds")
                time.sleep(0.1)
                waiting_cycle += 1

        raise TrafficLightException("Status Running was never release during the " + str(0.1*self.__max_wait_cycles) + " 1 minute waiting cycles")

    @synchronized_method
    def set_green_light(self):
        if not self.is_green_light():
            self.__change_color(GREEN)
            time.sleep(1)
            print("Light set to Green")
        else:
            print("Light is already Green")

    @synchronized_method
    def is_green_light(self):
        return self.__get_color() == GREEN

from aps.common.registry import GenericRegistry

@Singleton
class __TrafficLightRegistry(GenericRegistry):
    def __init__(self):
        GenericRegistry.__init__(self, registry_name="Traffic-Light")

    @synchronized_method
    def register_traffic_light(self, traffic_light_facade_instance, application_name=None): super().register_instance(traffic_light_facade_instance, application_name, False)

    @synchronized_method
    def reset(self, application_name=None): super().reset(application_name)

    @synchronized_method
    def get_traffic_light_instance(self, application_name=None): return super().get_instance(application_name)

# -----------------------------------------------------
# Factory Methods

def register_traffic_light_instance(reset=False, application_name=None, **kwargs):
    if reset: __TrafficLightRegistry.Instance().reset(application_name)
    __TrafficLightRegistry.Instance().register_traffic_light(__TrafficLightFile(application_name, **kwargs), application_name)

def get_registered_traffic_light_instance(application_name=None):
    return __TrafficLightRegistry.Instance().get_traffic_light_instance(application_name)
