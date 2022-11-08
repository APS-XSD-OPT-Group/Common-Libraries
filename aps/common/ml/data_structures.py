#!/usr/bin/env python
# -*- coding: utf-8 -*-
# #########################################################################
# Copyright (c) 2021, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2021. UChicago Argonne, LLC. This software was produced       #
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
# #########################################################################
import numpy

class DictionaryWrapper():
    def __init__(self, **kwargs):
        self.__dictionary = kwargs.copy()

    def set_parameter(self, parameter_name, parameter_value):
        self.__dictionary[parameter_name] = parameter_value

    def get_parameter(self, parameter_name):
        return self.__dictionary[parameter_name]

    def has_parameter(self, parameter_name):
        return parameter_name in self.__dictionary.keys()

    def get_parameter_names(self):
        return self.__dictionary.keys()

    def get_parameters_number(self):
        return len(self.__dictionary.keys())

    def names_to_numpy_array(self):
        return numpy.array(list(self.__dictionary.keys()))

    def values_to_numpy_array(self):
        return numpy.array(list(self.__dictionary.values()))

    def __str__(self):
        text = ""
        for parameter_name in self.get_parameter_names():
            parameter_value = self.get_parameter(parameter_name)

            if isinstance(parameter_value, str): text += parameter_value + "\n"
            else: text += parameter_name + ": " + str(self.get_parameter(parameter_name)) + "\n"

        return text

class ListOfParameters():
    def __init__(self):
        self.__list_of_parameters = []

    def add_parameters(self, parameters):
        if len(self.__list_of_parameters) > 0:
            if self.__list_of_parameters[0].get_parameters_number() != parameters.get_parameters_number():
                raise ValueError("Incompatible number of parameters")

        self.__list_of_parameters.append(parameters)

    def get_parameters(self, index):
        return self.__list_of_parameters[index]

    def get_number_of_parameters(self):
        return len(self.__list_of_parameters)

    def to_numpy_matrix(self):
        matrix = numpy.full(shape=(len(self.__list_of_parameters) + 1,
                                   self.__list_of_parameters[0].get_parameters_number()),
                            fill_value=None)
        matrix[0, :] = self.__list_of_parameters[0].names_to_numpy_array()

        for row_index in range(len(self.__list_of_parameters)):
            matrix[row_index + 1, :] = self.__list_of_parameters[row_index].values_to_numpy_array()

        return matrix

    def from_numpy_matrix(self, matrix):
        self.__list_of_parameters = []

        parameter_names = matrix[0, :]

        for row_index in range(1, matrix.shape[0]):
            input_parameters = DictionaryWrapper()
            for column_index in range(len(parameter_names)):
                input_parameters.set_parameter(parameter_names[column_index], matrix[row_index, column_index])
                self.__list_of_parameters.append(input_parameters)

    def to_npy_file(self, file_name):
        numpy.save(file_name, self.to_numpy_matrix(), allow_pickle=True)

    def from_npy_file(self, file_name):
        self.from_numpy_matrix(numpy.load(file_name, allow_pickle=True))

    def __str__(self):
        text = ""
        for parameters in self.__list_of_parameters: text += str(parameters) + "\n"

        return text
