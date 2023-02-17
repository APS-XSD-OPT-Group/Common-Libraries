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


from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QTextCursor

from aps.common.logger import LogStream, LoggerColor, LoggerHighlights, LoggerAttributes, _strip_colored_string
from aps.common.plot import gui

class LogStreamWidget(LogStream):
    class Widget(QWidget):
        def __init__(self, width, height, color):
            QWidget.__init__(self)

            self.__text_area_box = gui.widgetBox(self, "", orientation="vertical", height=height, width=width)

            self.__text_area = gui.textArea(readOnly=True)
            self.__text_area.setText("")
            self.__text_area.setStyleSheet("background-color: " + color)
            self.__text_area_box.layout().addWidget(self.__text_area)

            self.set_widget_size(width, height)

        def write(self, text : str):
            text, color, highlight, attrs = _strip_colored_string(text)

            if color is None: color = "#000000"
            else:
                if   color == LoggerColor.GRAY:    color = "#808080"
                elif color == LoggerColor.RED:     color = "#ff0000"
                elif color == LoggerColor.GREEN:   color = "#008000"
                elif color == LoggerColor.YELLOW:  color = "#ffff00"
                elif color == LoggerColor.BLUE:    color = "#0000ff"
                elif color == LoggerColor.MAGENTA: color = "#ff00ff"
                elif color == LoggerColor.CYAN:    color = "#00ffff"
                elif color == LoggerColor.WHITE:   color = "#ffffff"

            cursor = self.__text_area.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertHtml("<span style=\"color:" + color + ";\" >" + text + "</span>")
            cursor.insertText("\n")
            self.__text_area.setTextCursor(cursor)
            self.__text_area.ensureCursorVisible()

        def clear_log(self):
            self.__text_area.clear()

        def set_widget_size(self, width, height):
            self.setFixedWidth(width)
            self.setFixedHeight(height)
            self.__text_area_box.setFixedWidth(width)
            self.__text_area_box.setFixedHeight(height)
            self.__text_area.setFixedHeight(height - 5)
            self.__text_area.setFixedWidth(width - 5)

    def __init__(self, width=850, height=400, color='white'):
        self.__widget = LogStreamWidget.Widget(width, height, color)

    def close(self): pass
    def write(self, text): self.__widget.write(text)
    def flush(self, *args, **kwargs): pass
    def is_color_active(self): return True

    def get_widget(self):
        return self.__widget

    def set_widget_size(self, width=850, height=400):
        self.__widget.set_widget_size(width, height)
