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

class AlreadyInitializedError(ValueError):
    def __init__(self, message=None): super(AlreadyInitializedError, self).__init__(message)

class GenericRegistry(object):
    _NO_APPLICATION = "<NO APPLICATION>"

    def __init__(self, registry_name):
        self.__registry_name = registry_name
        self.__registry = {self._NO_APPLICATION: None}

    def register_instance(self, instance, application_name=None, replace=False):
        if instance is None: raise ValueError(self.__registry_name + " Instance is None")

        application_name = self.__get_application_name(application_name)

        if application_name in self.__registry.keys():
            if self.__registry[application_name] is None or replace==True: self.__registry[application_name] = instance
            else: raise AlreadyInitializedError(self.__registry_name + " Instance already initialized")
        else: self.__registry[application_name] = instance

    def reset(self, application_name=None):
        application_name = self.__get_application_name(application_name)

        if application_name in self.__registry.keys(): self.__registry[self.__get_application_name(application_name)] = None
        else: pass #raise ValueError(self.__registry_name + " Instance not existing")

    def get_instance(self, application_name=None):
        application_name = self.__get_application_name(application_name)

        if application_name in self.__registry.keys(): return self.__registry[self.__get_application_name(application_name)]
        else: raise ValueError(self.__registry_name + " Instance not existing")

    def __get_application_name(self, application_name):
        return self._NO_APPLICATION if application_name is None else application_name
