#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2023, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2023. UChicago Argonne, LLC. This software was produced       #
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
import importlib
import typing

def get_class_full_name(class_entity: typing.Type) -> str:
    module = class_entity.__module__
    if module is None or module == str.__class__.__module__: module = ""

    return module + '.' + class_entity.__qualname__

def class_for_name(module_name, class_name):
    try:
        module = importlib.import_module(module_name)
        try:  __class = getattr(module, class_name)
        except AttributeError: raise ValueError('Class ' + module_name + "." + class_name + ' does not exist')
    except ImportError: raise ValueError('Module ' + module_name + ' does not exist')

    return __class

def instance_for_name(module_name, class_name, **arguments):
    return class_for_name(module_name, class_name)(**arguments)

def method_of_module_for_name(module_name, method_name, **arguments):
    try: module = importlib.import_module(module_name)
    except ImportError: raise ValueError('Module ' + module_name + ' does not exist')

    try: return getattr(module, method_name)(**arguments)
    except AttributeError: raise ValueError('Method ' + method_name + ' does not exist')

def method_of_class_for_name(module_name, class_name, method_name, **arguments):
    __class = class_for_name(module_name, class_name)

    try: return getattr(__class, method_name)(**arguments)
    except AttributeError: raise ValueError('Method ' + method_name + ' does not exist')

def call_method_of_instance(instance, method_name, **arguments):
    try: return getattr(instance, method_name)(**arguments)
    except AttributeError: raise ValueError('Method ' + method_name + ' does not exist')

def get_class_name_and_module(full_name : str, raise_exception=True):
    try:
        tokens      = full_name.split(sep=".")
        class_name  = tokens[-1]
        if len(class_name) == 0: class_name = None
        module_name = full_name[:-(len(class_name)+1)]
        if len(module_name) == 0: module_name = None

        return class_name, module_name
    except:
        if raise_exception: raise ValueError("Malformed Class Name: " + full_name)
        else: return None, None

if __name__=="__main__":
    print(get_class_name_and_module("aps.common.ciccio.Pelliccio"))
    print(get_class_name_and_module("CiccioPelliccio"))
    print(get_class_name_and_module(None, raise_exception=False))
