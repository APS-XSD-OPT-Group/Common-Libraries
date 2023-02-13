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

from aps.common.plot import gui

from PyQt5.QtWidgets import QWidget, QDialog, QVBoxLayout, QHBoxLayout, QDialogButtonBox
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.pyplot import rcParams

pixels_to_inches = 1/rcParams['figure.dpi']

class AbstractGenericWidget(object):
    def __init__(self, application_name=None): self._application_name = application_name
    def build_widget(self, **kwargs): raise NotImplementedError()

class FigureToSave():
    def __init__(self, figure_file_name=None, figure=None):
        self.__figure_file_name = figure_file_name
        self.__figure           = figure

    def save_figure(self, **kwargs):
        if not self.__figure is None: self.__figure.savefig(self.__figure_file_name, **kwargs)

class GenericWidget(QWidget, AbstractGenericWidget):
    def __init__(self, parent=None, application_name=None, **kwargs):
        QWidget.__init__(self, parent=parent)
        AbstractGenericWidget.__init__(self, application_name=application_name)

        try:    self.__allows_saving             = kwargs["allows_saving"]
        except: self.__allows_saving             = True
        try:    self.__ignores_figure_dimensions = kwargs["ignores_figure_dimensions"]
        except: self.__ignores_figure_dimensions = False

    def get_plot_tab_name(self): raise NotImplementedError()

    def build_widget(self, **kwargs):
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        figure = self.build_mpl_figure(**kwargs)

        if not self._ignores_figure_dimensions():
            try:    figure_width  = kwargs["figure_width"]*pixels_to_inches
            except: figure_width  = figure.get_figwidth()
            try:    figure_height = kwargs["figure_height"]*pixels_to_inches
            except: figure_height = figure.get_figheight()

            figure.set_figwidth(figure_width)
            figure.set_figheight(figure_height)

        canvas = FigureCanvas(figure)
        canvas.setParent(self)

        self.append_mpl_figure_to_save(canvas.figure)

        try:    widget_width  = kwargs["widget_width"]
        except: widget_width  = canvas.get_width_height()[0]*1.1
        try:    widget_height = kwargs["widget_height"]
        except: widget_height = canvas.get_width_height()[1]*1.1

        self.setFixedWidth(widget_width)
        self.setFixedHeight(widget_height)

        layout.setStretchFactor(canvas, 1)
        layout.addWidget(canvas)

        self.setLayout(layout)

    def append_mpl_figure_to_save(self, figure, figure_file_name="Figure"):
        if not hasattr(self, "__figures_to_save") or self.__figures_to_save is None: self.__figures_to_save = []
        self.__figures_to_save.append(FigureToSave(figure=figure, figure_file_name=self._check_figure_file_name(figure_file_name)))

    def _check_figure_file_name(self, figure_file_name):
        return figure_file_name

    def build_mpl_figure(self, **kwargs): raise NotImplementedError()

    def _allows_saving(self): return self.__allows_saving
    def _ignores_figure_dimensions(self): return self.__ignores_figure_dimensions

    def get_figures_to_save(self):
        if self._allows_saving(): return self.__figures_to_save
        else: return None

class GenericInteractiveWidget(QDialog, AbstractGenericWidget):

    def __init__(self, parent, message, title, application_name=None, standard_buttons = [QDialogButtonBox.Ok, QDialogButtonBox.Cancel], **kwargs):
        QDialog.__init__(self, parent)
        AbstractGenericWidget.__init__(self, application_name=application_name)

        self.setWindowTitle(message)
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        self.__central_widget = gui.widgetBox(self, title, "vertical")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        standardButtons = None
        for sb in standard_buttons: standardButtons = sb if standardButtons is None else standardButtons | sb

        button_box = QDialogButtonBox(orientation=Qt.Horizontal, standardButtons=standardButtons)
        if QDialogButtonBox.Ok in standard_buttons:     button_box.accepted.connect(self.__accepted)
        if QDialogButtonBox.Cancel in standard_buttons: button_box.rejected.connect(self.__rejected)
        layout.addWidget(self.__central_widget)
        layout.addWidget(button_box)

        self.__output = None

    def __accepted(self):
        self.__output = self.get_accepted_output()
        self.accept()

    def __rejected(self):
        self.__output = self.get_rejected_output()
        self.reject()

    def get_output_object(self):
        return self.__output

    def get_accepted_output(self): raise NotImplementedError()
    def get_rejected_output(self): raise NotImplementedError()

    def get_central_widget(self):
        return self.__central_widget

    @classmethod
    def get_output(cls, dialog : QDialog):
        dialog.exec_()

        return dialog.get_output_object()
