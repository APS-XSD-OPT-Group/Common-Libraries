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

# CLASSES/METHODS to run OASYS offline
class MockWidget():
    def __init__(self, verbose=False,  workspace_units=2):
        self._verbose = verbose
        self._setWorkspaceUnits(workspace_units)

    def fixWeirdShadowBug(self):
        pass

    def set_verbose(self, verbose=False):
        self._verbose = verbose

    def is_verbose(self):
        return self._verbose==True

    def setStatusMessage(self, message):
        self.status_message(message)

    def progressBarSet(self, value):
        self.set_progress_bar(value)

    def status_message(self, message):
        if self._verbose: print(message)

    def set_progress_bar(self, value):
        if self._verbose: print("Mock Widget, operation completed at " + str(value) + " %")

    def _setWorkspaceUnits(self, units):
        self.workspace_units = units

        if self.workspace_units == 0:
            self.workspace_units_label = "m"
            self.workspace_units_to_m = 1.0
            self.workspace_units_to_cm = 100.0
            self.workspace_units_to_mm = 1000.0

        elif self.workspace_units == 1:
            self.workspace_units_label = "cm"
            self.workspace_units_to_m = 0.01
            self.workspace_units_to_cm = 1.0
            self.workspace_units_to_mm = 10.0

        elif self.workspace_units == 2:
            self.workspace_units_label = "mm"
            self.workspace_units_to_m = 0.001
            self.workspace_units_to_cm = 0.1
            self.workspace_units_to_mm = 1.0
