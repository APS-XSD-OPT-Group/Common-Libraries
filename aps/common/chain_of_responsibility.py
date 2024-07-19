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

from aps.common.reflection import instance_for_name, get_class_name_and_module
import traceback

class __AbstractChainOfResponsibility:
    def __init__(self): self._chain_of_responsibility = {}

    def initialize(self, classes : dict = {}, instances:dict = {}, **kwargs):
        for class_id in classes.keys():
            try:
                specific_class = classes.get(class_id)

                if specific_class is None or specific_class.strip()=="": raise ValueError("Warning on chain of responsibility initialization: skipped empty string")
                class_name, module_name = get_class_name_and_module(specific_class)

                if (class_name is None or class_name.strip() == "") or\
                   (module_name is None or module_name.strip() == ""): raise ValueError("Warning on chain of responsibility initialization: " + specific_class + " is specified with a wrong format")

                self._add_to_chain(class_id, class_name, module_name, **kwargs)
            except ValueError as e:
                print(e)
            except Exception:
                print(traceback.format_exc())

        for class_id in instances.keys():
            try:
                if not class_id in classes.keys():
                    instance = instances[class_id]
                    if not isinstance(instance, self._get_chain_interface()):
                        raise ValueError("Warning on chain of responsibility initialization: " + instance.__class__.__name__ +
                                         " is not of type " + self._get_chain_interface().__name__)
                    self._chain_of_responsibility[class_id] = instance
                else:
                    raise ValueError("Class " + class_id + " has been provided both as class name and as instance. Instance is ignored")
            except ValueError as e:
                print(e)
            except Exception:
                print(traceback.format_exc())

    def _add_to_chain(self, class_id : str, class_name : str, module_name : str, **kwargs): raise NotImplementedError()
    def _get_chain_interface(self) -> type: raise NotImplementedError()

    def get_instance_from_chain(self, class_id : str, **kwargs): raise NotImplementedError()

class DynamicChainOfResponsibility(__AbstractChainOfResponsibility):
    def _add_to_chain(self, class_id : str, class_name : str, module_name : str, **kwargs):
        self._chain_of_responsibility[class_id] = (module_name, class_name)

    def get_instance_from_chain(self, class_id : str = None, **kwargs):
        if class_id is None: raise ValueError("class id is None")

        try: item = self._chain_of_responsibility[class_id]
        except: raise ValueError("Class " + class_id + " not found")

        try:
            module_name, class_name = item

            instance = instance_for_name(module_name=module_name, class_name=class_name, **kwargs)

            if not isinstance(instance, self._get_chain_interface()):
                raise ValueError("Warning on chain of responsibility initialization: " + class_name +
                                 " is not of type " + self._get_chain_interface().__name__)
        except Exception as e:
            if kwargs.get("verbose", False):
                print("Warning, caught exception while instantiating: " + str(e))
                traceback.print_exc()

            instance = item

        return instance

class StaticChainOfResponsibility(__AbstractChainOfResponsibility):
    def _add_to_chain(self, class_id : str, class_name : str, module_name : str, **kwargs):
        instance = instance_for_name(module_name=module_name, class_name=class_name, **kwargs)

        if not isinstance(instance, self._get_chain_interface()):
            raise ValueError("Warning on chain of responsibility initialization: " + class_name +
                             " is not of type " + self._get_chain_interface().__name__)

        self._chain_of_responsibility[class_id] = instance

    def get_instance_from_chain(self, class_id : str = None, **kwargs):
        if class_id is None: raise ValueError("class id is None")
        if len(kwargs.keys()) > 0: print("kwargs are ignored by this method: instance are already stored during initialization")

        try: instance = self._chain_of_responsibility[class_id]
        except: raise ValueError("Class " + class_id + " not found")

        return instance
