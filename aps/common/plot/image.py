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
import numpy

def get_fwhm(histogram, bins):
    quote = numpy.max(histogram)*0.5
    cursor = numpy.where(histogram >= quote)

    if histogram[cursor].size > 1:
        bin_size    = bins[1]-bins[0]
        fwhm        = bin_size*(cursor[0][-1]-cursor[0][0])
        coordinates = (bins[cursor[0][0]], bins[cursor[0][-1]])
    else:
        fwhm = 0.0
        coordinates = None

    return fwhm, quote, coordinates

def get_sigma(histogram, bins):
    frequency = histogram/numpy.sum(histogram)
    average   = numpy.sum(frequency*bins)
    return numpy.sqrt(numpy.sum(frequency*((bins-average)**2)))

def get_rms(histogram, bins):
    frequency = histogram/numpy.sum(histogram)
    return numpy.sqrt(numpy.sum(frequency*(bins**2)))

def get_average(histogram, bins):
    frequency = histogram/numpy.sum(histogram)
    return numpy.sum(frequency*bins)

def get_peak_location(histogram, bins):
    return bins[numpy.argmax(histogram)]

from scipy.ndimage import median_filter

def get_peak_location_2D(x_array, y_array, z_array, smooth=False):
    if smooth: z_array = median_filter(z_array, size=3)
    indexes = numpy.unravel_index(numpy.argmax(z_array, axis=None), z_array.shape)

    return x_array[indexes[0]], numpy.flip(y_array)[indexes[1]]