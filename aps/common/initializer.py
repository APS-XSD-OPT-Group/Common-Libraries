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
import json
from typing import Any, List, Optional, Type

import os
from configparser import ConfigParser

class IniMode:
    LOCAL_FILE = 0
    REMOTE_FILE = 1
    DATABASE = 2
    LOCAL_JSON_FILE = 3
    REMOTE_JSON_FILE = 4
    NONE = 99

class IniFacade:
    def set_value_at_ini(self, section, key, value): raise NotImplementedError()
    def set_list_at_ini(self, section, key, values_list=[]): raise NotImplementedError()
    def get_string_from_ini(self, section, key, default=None): raise NotImplementedError()
    def get_int_from_ini(self, section, key, default=None): raise NotImplementedError()
    def get_float_from_ini(self, section, key, default=None): raise NotImplementedError()
    def get_boolean_from_ini(self, section, key, default=False): raise NotImplementedError()
    def get_list_from_ini(self, section, key, default=None, type=str): raise NotImplementedError()
    def dump(self): raise NotImplementedError()
    def push(self): raise NotImplementedError()

class __NullIni(IniFacade):
    def set_value_at_ini(self, section, key, value): pass
    def set_list_at_ini(self, section, key, values_list=[]): pass
    def get_string_from_ini(self, section, key, default=None): pass
    def get_int_from_ini(self, section, key, default=None): pass
    def get_float_from_ini(self, section, key, default=None): pass
    def get_boolean_from_ini(self, section, key, default=False): pass
    def get_list_from_ini(self, section, key, default=None, type=str): pass
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

    def get_list_from_ini(self, section, key, default=[], type=str):
        value = self.__get_from_ini(section, key, default=None)
        if value is None: return default
        else:
            values = value.split(',')
            values = [value.strip() for value in values]

            if   type==int:
                try:    return [int(value) for value in values]
                except: return []
            elif type==float:
                try: return [float(value) for value in values]
                except: return []
            elif type==bool:
                try: return [(True if value.lower() == "true" else False) for value in values]
                except: return []
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

class __LocalJsonFile(IniFacade):
    def __init__(self, **kwargs):
        self.__json_file_name = kwargs["json_file_name"]

        try: self.__verbose       = kwargs["verbose"]
        except: self.__verbose = True

        if not os.path.isfile(self.__json_file_name):
            with open(self.__json_file_name, "w") as json_file: json_file.write('\n')
            if self.__verbose: print("File " + self.__json_file_name + " doesn't exist: created empty json file.")

        try:
            with open(self.__json_file_name, 'r') as file: self.__data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            self.__data = {}

    def __get_nested_key(self, section, keys, default, raise_key_error=False):
        try:
            data = self.__data[section]
            for key in keys[:-1]: data = data[key]
            return data.get(keys[-1], default)
        except KeyError:
            if raise_key_error:
                raise KeyError(f"Key path {section} {' -> '.join(keys)} not found.")
            else:
                self.__set_nested_key(section, keys, default)
                return default

    def __set_nested_key(self, section, keys, value):
        try:
            try:
                data = self.__data[section]
            except:
                self.__data[section] = {}
                data = self.__data[section]
            for key in keys[:-1]:
                if key in data and not isinstance(data[key], dict): raise ValueError(f"Incompatibile key path: {key} is not nested")
                if key not in data: data[key] = {}
                data = data[key]
            data[keys[-1]] = value
        except Exception as e:
            raise KeyError(f"Failed to add or update key path {' -> '.join(keys)}: {e}")

    def set_value_at_ini(self, section: str, key: str, value: Any):
        self.__set_nested_key(section, key.split(','), value)

    def set_list_at_ini(self, section: str, key: str, values_list: List[Any] = []):
        self.__set_nested_key(section, key.split(','), values_list)

    def get_string_from_ini(self, section: str, key: str, default: Optional[str] = None) -> Optional[str]:
        return str(self.__get_nested_key(section, key.split(','), default))

    def get_int_from_ini(self, section: str, key: str, default: Optional[int] = None) -> Optional[int]:
        try: return int(self.__get_nested_key(section, key.split(','), default))
        except (ValueError, TypeError): return default

    def get_float_from_ini(self, section: str, key: str, default: Optional[float] = None) -> Optional[float]:
        try: return float(self.__get_nested_key(section, key.split(','), default))
        except (ValueError, TypeError): return default

    def get_boolean_from_ini(self, section: str, key: str, default: bool = False) -> bool:
        value = self.__get_nested_key(section, key, False if default is None else default)
        if isinstance(value, str): return value.lower() in ('true', '1', 'yes')
        return bool(value)

    def get_list_from_ini(self, section: str, key: str, default: Optional[List[Any]] = None, type: Type = str) -> List[Any]:
        value = self.__get_nested_key(section, key.split(','), default)
        if not isinstance(value, list): return default or []
        if type == bool: return [(item if type(item) == bool else (True if (type(item)==str and item.strip().lower() in ('true', '1', 'yes')) else False))  for item in value]
        else:            return [type(item) for item in value]

    def dump(self):
        return "Dump of file: " + self.__json_file_name + "\n" + \
               "%============================================================\n" + \
               json.dumps(self.__data, indent=4) + \
               "\n%============================================================\n"

    def push(self):
        with open(self.__json_file_name, 'w') as file: json.dump(self.__data, file, indent=4)


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
    if ini_mode == IniMode.LOCAL_JSON_FILE: __IniRegistry.Instance().register_ini(__LocalJsonFile(**kwargs), application_name)
    elif ini_mode == IniMode.NONE:     __IniRegistry.Instance().register_ini(__NullIni(), application_name)

def get_registered_ini_instance(application_name=None) -> IniFacade:
    return __IniRegistry.Instance().get_ini_instance(application_name)

if __name__=="__main__":
    import tempfile

    temp_dir = tempfile.gettempdir()
    print(temp_dir)

    register_ini_instance(IniMode.LOCAL_JSON_FILE, application_name="GIGIO", json_file_name=os.path.join(temp_dir, "test_ini.json"))

    ini = get_registered_ini_instance("GIGIO")

    values_list = ini.get_list_from_ini("Section1", "key1, key2, key3", type=int)

    ini.set_value_at_ini("Section1", "key1, key2, key3", [x + 1 for x in values_list])

    print(ini.dump())

    ini.push()

    print(ini.get_list_from_ini("Section1", "key1, key2, key3", type=int))

