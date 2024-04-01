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
import os
import numbers

from aps.common.plot.gui import ConfirmDialog

def check_num(num):
    if isinstance(num, str): return num.lower().replace('.','', 1).replace('e', '', 1).replace('-', '', 1).replace('+', '', 1).isdigit()
    else:                    return isinstance(num, numbers.Number)

def check_positive(num, strictly=True):
    if strictly: return check_num(num) and float(num) > 0.0
    else:        return check_num(num) and float(num) >= 0.0

def check_int(num):  return check_num(num) and type(num) == int
def check_sign(num): return check_num(num) and check_int(num) and num in [-1, 0, 1]

def check_range_boundaries(rb_1, rb_2, is_scale=True):
    if check_num(rb_1) and check_num(rb_2):
        if float(rb_1) == 0.0 and float(rb_2) == 0.0: return is_scale
        else:                                         return rb_1 < rb_2
    else: return False

def check_path_existance(path, name):
    if not os.path.exists(path): raise ValueError(name + " does not exist")


from PyQt5.QtWidgets import QLabel

def check_and_create_directory(parent, folder, folder_name):
    if not os.path.exists(folder):
        row_1 = "<p align='left'>" + folder_name + " does not exist. Do you confirm creation of a new folder with the following path:<br><br>"
        row_2 = folder + "</p>"

        size_1 = QLabel().fontMetrics().boundingRect(row_1)
        size_2 = QLabel().fontMetrics().boundingRect(row_2)

        if ConfirmDialog.confirmed(parent, title="Create " + folder_name,
                                   message=row_1 + row_2,
                                   width=max(size_1.width(), size_2.width()), height=130+size_1.height()*4):
            try: os.mkdir(folder)
            except Exception as e: raise ValueError("Exception occurred while creating " + folder_name + ":\n"+ str(e.args[0]))
        else: raise ValueError("Creation of " + folder_name + " aborted")

def check_empty_string(string):
    if not string is None: return string.strip() != ""
    else: return False
