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

from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QWidget
from PyQt5.QtCore import QRect

from aps.common.plot.gui import stylesheet_string

##########################################################################
# WIDGET FOR SCRIPTING

class AbstractContextWidget():
    def __init__(self, container_widget):
        self.__container_widget = container_widget

    def get_container_widget(self):
        return self.__container_widget

class DefaultContextWidget(AbstractContextWidget):
    def __init__(self, container_widget):
        super(DefaultContextWidget, self).__init__(container_widget)

class DefaultMainWindow(QMainWindow, AbstractContextWidget):
    def __init__(self, title):
        super(DefaultMainWindow, self).__init__(container_widget=QWidget())
        self.setWindowTitle(title)
        self.setCentralWidget(self.get_container_widget())

        desktop_widget = QDesktopWidget()
        actual_geometry = self.frameGeometry()
        screen_geometry = desktop_widget.availableGeometry()
        new_geometry = QRect()
        new_geometry.setWidth(actual_geometry.width())
        new_geometry.setHeight(actual_geometry.height())
        new_geometry.setTop(screen_geometry.height()*0.05)
        new_geometry.setLeft(screen_geometry.width()*0.05)

        self.setGeometry(new_geometry)

        self.setStyleSheet(stylesheet_string)

class PlottingProperties:
    def __init__(self, container_widget=None, context_widget=None, **parameters):
        self.__container_widget = container_widget
        self.__context_widget = context_widget
        self.__parameters = parameters

    def get_container_widget(self):
        return self.__container_widget

    def get_context_widget(self):
        return self.__context_widget

    def get_parameters(self):
        return self.__parameters

    def get_parameter(self, parameter_name, default_value=None):
        try:
            return self.__parameters[parameter_name]
        except:
            return default_value

    def set_parameter(self, parameter_name, value):
        self.__parameters[parameter_name] = value


WIDGET_FIXED_WIDTH = 800
