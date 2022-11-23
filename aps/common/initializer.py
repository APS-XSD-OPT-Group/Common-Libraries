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

from aps.common.singleton import Singleton, synchronized_method

import os
from configparser import ConfigParser

class IniMode:
    LOCAL_FILE = 0
    REMOTE_FILE = 1
    DATABASE = 2
    NONE = 99

class IniFacade:
    def set_value_at_ini(self, section, key, value): raise NotImplementedError()
    def set_list_at_ini(self, section, key, values_list=[]): raise NotImplementedError()
    def get_string_from_ini(self, section, key, default=None): raise NotImplementedError()
    def get_int_from_ini(self, section, key, default=None): raise NotImplementedError()
    def get_float_from_ini(self, section, key, default=None): raise NotImplementedError()
    def get_boolean_from_ini(self, section, key, default=False): raise NotImplementedError()
    def get_list_from_ini(self, section, key, default=None): raise NotImplementedError()
    def dump(self): raise NotImplementedError()
    def push(self): raise NotImplementedError()

class __NullIni(IniFacade):
    def set_value_at_ini(self, section, key, value): pass
    def set_list_at_ini(self, section, key, values_list=[]): pass
    def get_string_from_ini(self, section, key, default=None): pass
    def get_int_from_ini(self, section, key, default=None): pass
    def get_float_from_ini(self, section, key, default=None): pass
    def get_boolean_from_ini(self, section, key, default=False): pass
    def get_list_from_ini(self, section, key, default=None): pass
    def dump(self): pass
    def push(self): pass

class __LocalIniFile(IniFacade):
    def __init__(self, **kwargs):
        self.__ini_file_name = kwargs["ini_file_name"]
        try: self.__verbose       = kwargs["verbose"]
        except: self.__verbose = True

        if not os.path.isfile(self.__ini_file_name):
            with open(self.__ini_file_name, "w") as ini_file: ini_file.write('\n')
            if self.__verbose: print("File " + self.__ini_file_name + " doesn't exist: created empty ini file.")

        self.__config_parser = ConfigParser()
        self.__config_parser.read(self.__ini_file_name)

    def get_ini_file_name(self):
        return self.__ini_file_name

    def __get_from_ini(self, section, key, default=None):
        try:
            value = self.__config_parser[section][key]
            value = value.strip()
            return None if value.lower() == "none" else value
        except:
            return str(default) if not default is None else None

    def set_value_at_ini(self, section, key, value):
        try:
            self.__config_parser[section][key] = "None" if value is None else str(value)
        except:
            if not self.__config_parser.has_section(section): self.__config_parser.add_section(section)
            if not self.__config_parser.has_option(section, key): self.__config_parser.set(section, key, "None" if value is None else str(value))

    def set_list_at_ini(self, section, key, values_list=[]):
        if values_list is None: values_string = "None"
        else:
            values_string = ""
            for value in values_list:
                if type(value) == str: values_string += value + ", "
                else:                  values_string += str(value) + ", "
            values_string = values_string[:-2]

        self.set_value_at_ini(section, key, values_string)

    def get_string_from_ini(self, section, key, default="None"):
        value = self.__get_from_ini(section, key, default)
        return (str(default) if not default is None else None) if value is None else value.strip()

    def get_int_from_ini(self, section, key, default=0):
        value = self.__get_from_ini(section, key, default)
        return (int(default) if not default is None else None) if value is None else int(value.strip())

    def get_float_from_ini(self, section, key, default=0.0):
        value = self.__get_from_ini(section, key, default)
        return (float(default)  if not default is None else None) if value is None else float(value.strip())

    def get_boolean_from_ini(self, section, key, default=False):
        value = self.__get_from_ini(section, key, default)
        return (default if not default is None else False) if value is None else (True if value.strip().lower() == "true" else False)

    def get_list_from_ini(self, section, key, default=[], type=int):
        value = self.__get_from_ini(section, key, default=None)
        if value is None: return default
        else:
            values = self.__get_from_ini(section, key).split(',')
            values = [value.strip() for value in values]
            if   type==int:   return [int(value) for value in values]
            elif type==float: return [float(value) for value in values]
            elif type==bool:  return [(True if value.lower() == "true" else False) for value in values]
            elif type==str:   return values
            else: raise ValueError("type not recognized")

    def dump(self):
        text = "Dump of file: " + self.__ini_file_name + "\n" + \
               "%============================================================"

        for section in self.__config_parser.sections():
            text += "\n[" + section + "]\n"

            for option in self.__config_parser.options(section):
                text += option + " = " + str(self.__config_parser.get(section, option)) + "\n"

        text += "%============================================================\n"

        return text

    def push(self):
        with open(self.__ini_file_name, "w") as ini_file: self.__config_parser.write(ini_file)

from aps.common.registry import GenericRegistry

@Singleton
class __IniRegistry(GenericRegistry):
    def __init__(self):
        GenericRegistry.__init__(self, registry_name="Ini")

    @synchronized_method
    def register_ini(self, ini_facade_instance, application_name=None):
        super().register_instance(ini_facade_instance, application_name, False)

    @synchronized_method
    def reset(self, application_name=None):
        super().reset(application_name)

    def get_ini_instance(self, application_name=None):
        return super().get_instance(application_name)


# -----------------------------------------------------
# Factory Methods

def register_ini_instance(ini_mode=IniMode.LOCAL_FILE, reset=False, application_name=None, **kwargs):
    if reset: __IniRegistry.Instance().reset(application_name)
    if ini_mode == IniMode.LOCAL_FILE: __IniRegistry.Instance().register_ini(__LocalIniFile(**kwargs), application_name)
    elif ini_mode == IniMode.NONE:     __IniRegistry.Instance().register_ini(__NullIni(), application_name)

def get_registered_ini_instance(application_name=None):
    return __IniRegistry.Instance().get_ini_instance(application_name)



