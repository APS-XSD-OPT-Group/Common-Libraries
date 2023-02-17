# #########################################################################
# Copyright (c) 2020, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2020. UChicago Argonne, LLC. This software was produced       #
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
from aps.common.singleton import Singleton, synchronized_method
import sys, io, numpy

DEFAULT_STREAM=sys.stdout

class LogStream(io.TextIOWrapper):
    def close(self, *args, **kwargs): raise NotImplementedError()
    def flush(self, *args, **kwargs): raise NotImplementedError()
    def write(self, *args, **kwargs): raise NotImplementedError()
    def is_color_active(self): return False

class LoggerFacade:
    def print(self, message): raise NotImplementedError()
    def print_message(self, message): raise NotImplementedError()
    def print_warning(self, message): raise NotImplementedError()
    def print_error(self, message): raise NotImplementedError()
    def print_other(self, message, prefix, color): raise NotImplementedError()

class LoggerMode:
    FULL = 0
    WARNING = 1
    ERROR = 2
    NONE = 3
    
    @classmethod
    def get_logger_mode(cls, logger_mode=FULL):
        if logger_mode==cls.FULL: return "Full" 
        if logger_mode==cls.WARNING: return "Warning" 
        if logger_mode==cls.ERROR: return "Error" 
        if logger_mode==cls.NONE: return "None" 

class LoggerColor:
    GRAY    = "grey"
    RED     = "red"
    GREEN   = "green"
    YELLOW  = "yellow"
    BLUE    = "blue"
    MAGENTA = "magenta"
    CYAN    = "cyan"
    WHITE   = "white"

class LoggerHighlights:
    NONE       = None
    ON_GREY    = "on_grey"
    ON_RED     = "on_red"
    ON_GREEN   = "on_green"
    ON_YELLOW  = "on_yellow"
    ON_BLUE    = "on_blue"
    ON_MAGENTA = "on_magenta"
    ON_CYAN    = "on_cyan"
    ON_WHITE   = "on_white"

class LoggerAttributes:
    NONE      = None
    BOLD      = "bold"
    DARK      = "dark"
    UNDERLINE = "underline"
    BLINK     = "blink"
    REVERSE   = "reverse"
    CONCEALED = "concealed"

class _InnerColors:
    END = "\033[00m"

    BOLD          = "\033[1m"
    DARK          = "\033[2m"
    UNDERLINE     = "\033[4m"
    BLINK         = "\033[5m"
    REVERSE       = "\033[7m"
    CONCEALED     = "\033[8m"

    RESET_BOLD      = "\033[21m"
    RESET_DARK      = "\033[22m"
    RESET_UNDERLINE = "\033[24m"
    RESET_BLINK     = "\033[25m"
    RESET_REVERSE   = "\033[27m"
    RESET_CONCEALED = "\033[28m"

    DEFAULT      = "\033[39m"
    BLACK        = "\033[30m"
    RED          = "\033[31m"
    GREEN        = "\033[32m"
    YELLOW       = "\033[33m"
    BLUE         = "\033[34m"
    MAGENTA      = "\033[35m"
    CYAN         = "\033[36m"
    LIGHTGRAY    = "\033[37m"
    DARKGRAY     = "\033[90m"
    LIGHTRED     = "\033[91m"
    LIGHTGREEN   = "\033[92m"
    LIGHTYELLOW  = "\033[93m"
    LIGHTBLUE    = "\033[94m"
    LIGHTMAGENTA = "\033[95m"
    LIGHTCYAN    = "\033[96m"
    WHITE        = "\033[97m"

    BACKGROUND_DEFAULT      = "\033[49m"
    BACKGROUND_BLACK        = "\033[40m"
    BACKGROUND_RED          = "\033[41m"
    BACKGROUND_GREEN        = "\033[42m"
    BACKGROUND_YELLOW       = "\033[43m"
    BACKGROUND_BLUE         = "\033[44m"
    BACKGROUND_MAGENTA      = "\033[45m"
    BACKGROUND_CYAN         = "\033[46m"
    BACKGROUND_LIGHTGRAY    = "\033[47m"
    BACKGROUND_DARKGRAY     = "\033[100m"
    BACKGROUND_LIGHTRED     = "\033[101m"
    BACKGROUND_LIGHTGREEN   = "\033[102m"
    BACKGROUND_LIGHTYELLOW  = "\033[103m"
    BACKGROUND_LIGHTBLUE    = "\033[104m"
    BACKGROUND_LIGHTMAGENTA = "\033[105m"
    BACKGROUND_LIGHTCYAN    = "\033[106m"
    BACKGROUND_WHITE        = "\033[107m"


def _strip_colored_string(text):
    color = None
    highlight = None
    attrs = None

    if _InnerColors.END[1:] == text[-4:]:
        text = text[:-5]

        has_attributes = True
        if   text [1:4] == _InnerColors.BOLD[1:]      : attrs = LoggerAttributes.BOLD
        elif text [1:4] == _InnerColors.DARK[1:]      : attrs = LoggerAttributes.DARK
        elif text [1:4] == _InnerColors.UNDERLINE[1:] : attrs = LoggerAttributes.UNDERLINE
        elif text [1:4] == _InnerColors.BLINK[1:]     : attrs = LoggerAttributes.BLINK
        elif text [1:4] == _InnerColors.REVERSE[1:]   : attrs = LoggerAttributes.REVERSE
        elif text [1:4] == _InnerColors.CONCEALED[1:] : attrs = LoggerAttributes.CONCEALED
        else: has_attributes = False
        if has_attributes == True: text = text[4:]

        has_highlights = True
        if   text [1:6] == _InnerColors.BACKGROUND_DARKGRAY[1:] : highlight = LoggerHighlights.ON_GREY
        elif text [1:5] == _InnerColors.BACKGROUND_RED[1:]      : highlight = LoggerHighlights.ON_RED
        elif text [1:5] == _InnerColors.BACKGROUND_GREEN[1:]    : highlight = LoggerHighlights.ON_GREEN
        elif text [1:5] == _InnerColors.BACKGROUND_YELLOW[1:]   : highlight = LoggerHighlights.ON_YELLOW
        elif text [1:5] == _InnerColors.BACKGROUND_BLUE[1:]     : highlight = LoggerHighlights.ON_BLUE
        elif text [1:5] == _InnerColors.BACKGROUND_MAGENTA[1:]  : highlight = LoggerHighlights.ON_MAGENTA
        elif text [1:5] == _InnerColors.BACKGROUND_CYAN[1:]     : highlight = LoggerHighlights.ON_CYAN
        elif text [1:5] == _InnerColors.BACKGROUND_WHITE[1:]    : highlight = LoggerHighlights.ON_WHITE
        else: has_highlights = False
        if has_highlights == True: text = text[6:] if highlight == LoggerHighlights.ON_GREY else text[5:]

        has_color = True
        if   text [1:5] == _InnerColors.DARKGRAY[1:] : color = LoggerColor.GRAY
        elif text [1:5] == _InnerColors.RED[1:]      : color = LoggerColor.RED
        elif text [1:5] == _InnerColors.GREEN[1:]    : color = LoggerColor.GREEN
        elif text [1:5] == _InnerColors.YELLOW[1:]   : color = LoggerColor.YELLOW
        elif text [1:5] == _InnerColors.BLUE[1:]     : color = LoggerColor.BLUE
        elif text [1:5] == _InnerColors.MAGENTA[1:]  : color = LoggerColor.MAGENTA
        elif text [1:5] == _InnerColors.CYAN[1:]     : color = LoggerColor.CYAN
        elif text [1:5] == _InnerColors.WHITE[1:]    : color = LoggerColor.WHITE
        else: has_color = False

        if has_color == True: text = text[5:]

        return text, color, highlight, attrs
    else:
        return text, None, None, None

def _get_colored_string(text, color, highlight, attrs):
    changed_color     = True
    changed_highlight = True
    changed_attrs     = True

    if   color == LoggerColor.GRAY:    text = _InnerColors.DARKGRAY + text
    elif color == LoggerColor.RED:     text = _InnerColors.RED      + text
    elif color == LoggerColor.GREEN:   text = _InnerColors.GREEN    + text
    elif color == LoggerColor.YELLOW:  text = _InnerColors.YELLOW   + text
    elif color == LoggerColor.BLUE:    text = _InnerColors.BLUE     + text
    elif color == LoggerColor.MAGENTA: text = _InnerColors.MAGENTA  + text
    elif color == LoggerColor.CYAN:    text = _InnerColors.CYAN     + text
    elif color == LoggerColor.WHITE:   text = _InnerColors.WHITE    + text
    else:                              changed_color = False

    if   highlight == LoggerHighlights.NONE:       changed_highlight = False
    elif highlight == LoggerHighlights.ON_GREY:    text = _InnerColors.BACKGROUND_DARKGRAY + text
    elif highlight == LoggerHighlights.ON_RED:     text = _InnerColors.BACKGROUND_RED      + text
    elif highlight == LoggerHighlights.ON_GREEN:   text = _InnerColors.BACKGROUND_GREEN    + text
    elif highlight == LoggerHighlights.ON_YELLOW:  text = _InnerColors.BACKGROUND_YELLOW   + text
    elif highlight == LoggerHighlights.ON_BLUE:    text = _InnerColors.BACKGROUND_BLUE     + text
    elif highlight == LoggerHighlights.ON_MAGENTA: text = _InnerColors.BACKGROUND_MAGENTA  + text
    elif highlight == LoggerHighlights.ON_CYAN:    text = _InnerColors.BACKGROUND_CYAN     + text
    elif highlight == LoggerHighlights.ON_WHITE:   text = _InnerColors.BACKGROUND_WHITE    + text

    if   attrs == LoggerAttributes.NONE:      changed_attrs = False
    elif attrs == LoggerAttributes.BOLD:      text = _InnerColors.BOLD      + text
    elif attrs == LoggerAttributes.DARK:      text = _InnerColors.DARK      + text
    elif attrs == LoggerAttributes.UNDERLINE: text = _InnerColors.UNDERLINE + text
    elif attrs == LoggerAttributes.BLINK:     text = _InnerColors.BLINK     + text
    elif attrs == LoggerAttributes.REVERSE:   text = _InnerColors.REVERSE   + text
    elif attrs == LoggerAttributes.CONCEALED: text = _InnerColors.CONCEALED + text

    if changed_color or changed_highlight or changed_attrs: text += _InnerColors.END
    else:                                                   text += "\n"

    return text

import platform

class __FullLogger(LoggerFacade):
    def __init__(self, stream=DEFAULT_STREAM):
        self.__stream = stream

        if platform.system() == 'Windows':
            self.__color_active = False
        else:
            if stream == DEFAULT_STREAM:
                self.__color_active = True
            elif isinstance(stream, LogStream):
                self.__color_active = stream.is_color_active()
            else:
                self.__color_active = False

    def print(self, message):
        self.__stream.write(message + "\n")
        self.__stream.flush()

    def __print_color(self, message, color=LoggerColor.GRAY, highlights=LoggerHighlights.NONE, attrs=LoggerAttributes.NONE):
        self.__stream.write(_get_colored_string(message, color, highlights, attrs) if self.__color_active else (message + "\n"))
        self.__stream.flush()

    def print_other(self, message, prefix="", color=LoggerColor.GRAY):
        self.__print_color(str(prefix) + str(message), color=color)

    def print_message(self, message):
        self.__print_color("MESSAGE: " + str(message), color=LoggerColor.BLUE)

    def print_warning(self, message):
        self.__print_color("WARNING: " + str(message), color=LoggerColor.MAGENTA)

    def print_error(self, message):
        self.__print_color("ERROR: " + str(message), color=LoggerColor.RED, highlights=LoggerHighlights.ON_GREEN, attrs=[LoggerAttributes.BOLD, LoggerAttributes.BLINK])

class __NullLogger(LoggerFacade):
    def __init__(self, stream=DEFAULT_STREAM): pass
    def print(self, message): pass
    def print_message(self, message): pass
    def print_warning(self, message): pass
    def print_error(self, message): pass
    def print_other(self, message, prefix, color): pass

class __ErrorLogger(__FullLogger):
    def __init__(self, stream=DEFAULT_STREAM): super().__init__(stream)
    def print(self, message): pass
    def print_message(self, message): pass
    def print_warning(self, message): pass
    def print_other(self, message, prefix, color): pass

class __WarningLogger(__FullLogger):
    def __init__(self, stream=DEFAULT_STREAM): super().__init__(stream)
    def print(self, message): pass
    def print_message(self, message): pass
    def print_other(self, message, prefix, color): pass

class __LoggerPool(LoggerFacade):
    def __init__(self, logger_list):
        if logger_list is None: raise ValueError("Logger list is None")
        for logger in logger_list:
            if not isinstance(logger, LoggerFacade): raise ValueError("Wrong objects in Logger list")

        self.__logger_list = numpy.array(logger_list)

    def print(self, message):
        for logger in self.__logger_list:
            logger.print(message)

    def print_message(self, message):
        for logger in self.__logger_list:
            logger.print_message(message)

    def print_warning(self, message):
        for logger in self.__logger_list:
            logger.print_warning(message)

    def print_error(self, message):
        for logger in self.__logger_list:
            logger.print_error(message)


class __AbstractLoggerRegistry:
    _NO_APPLICATION = "<NO APPLICATION>"

    def register_logger(self, logger_facade_instance, application_name=None): raise NotImplementedError()
    def reset(self, application_name=None): NotImplementedError()
    def get_logger_instance(self, application_name=None): NotImplementedError()

    def _get_application_name(self, application_name):
        return self._NO_APPLICATION if application_name is None else application_name

from aps.common.registry import GenericRegistry

@Singleton
class __LoggerRegistry(__AbstractLoggerRegistry, GenericRegistry):
    def __init__(self):
        GenericRegistry.__init__(self, registry_name="Logger")

    @synchronized_method
    def register_logger(self, logger_facade_instance, application_name=None, replace=False):
        super().register_instance(logger_facade_instance, application_name, replace)

    @synchronized_method
    def reset(self, application_name=None):
        super().reset(application_name)

    @synchronized_method
    def get_logger_instance(self, application_name=None):
        return super().get_instance(application_name)


SECONDARY_LOGGER = "Secondary_Logger"

@Singleton
class __SecondaryLoggerRegistry(__AbstractLoggerRegistry):
    def __init__(self):
        self.__logger_instances = {self._NO_APPLICATION: {}}

    @synchronized_method
    def register_logger(self, logger_facade_instance, logger_name=SECONDARY_LOGGER, application_name=None):
        if logger_facade_instance is None: raise ValueError("Logger Instance is None")
        if not isinstance(logger_facade_instance, LoggerFacade): raise ValueError("Logger Instance do not implement Logger Facade")

        application_name = self._get_application_name(application_name)

        if application_name in self.__logger_instances.keys():
            if self.__logger_instances[application_name] is None: self.__logger_instances[application_name] = {logger_name : logger_facade_instance}
            else: self.__logger_instances[application_name][logger_name] = logger_facade_instance
        else:
            self.__logger_instances[application_name] = {logger_name : logger_facade_instance}

    @synchronized_method
    def reset(self, application_name=None):
        application_name = self._get_application_name(application_name)

        if application_name in self.__logger_instances.keys(): self.__logger_instances[self._get_application_name(application_name)] = {}
        else: raise ValueError("Logger Instance not existing")

    @synchronized_method
    def get_logger_instance(self, logger_name=SECONDARY_LOGGER, application_name=None):
        application_name = self._get_application_name(application_name)

        if application_name in self.__logger_instances.keys():
            logger_instances = self.__logger_instances[self._get_application_name(application_name)]
            if logger_name in logger_instances.keys(): return logger_instances[logger_name]
            else: raise ValueError("Logger Instance not existing in the application registry")
        else:
            raise ValueError("Logger Instance not existing (application not registered)")

# -----------------------------------------------------
# Factory Methods

def register_logger_pool_instance(stream_list=[], logger_mode=LoggerMode.FULL, reset=False, application_name=None, replace=False):
    if reset: __LoggerRegistry.Instance().reset()
    if logger_mode==LoggerMode.FULL:      logger_list = [__FullLogger(stream) for stream in stream_list]
    elif logger_mode==LoggerMode.NONE:    logger_list = [__NullLogger(stream) for stream in stream_list]
    elif logger_mode==LoggerMode.WARNING: logger_list = [__WarningLogger(stream) for stream in stream_list]
    elif logger_mode==LoggerMode.ERROR:   logger_list = [__ErrorLogger(stream) for stream in stream_list]

    __LoggerRegistry.Instance().register_logger(__LoggerPool(logger_list=logger_list), application_name, replace)

def register_logger_single_instance(stream=DEFAULT_STREAM, logger_mode=LoggerMode.FULL, reset=False, application_name=None, replace=False):
    if reset: __LoggerRegistry.Instance().reset(application_name)
    if logger_mode==LoggerMode.FULL:      __LoggerRegistry.Instance().register_logger(__FullLogger(stream), application_name, replace)
    elif logger_mode==LoggerMode.NONE:    __LoggerRegistry.Instance().register_logger(__NullLogger(stream), application_name, replace)
    elif logger_mode==LoggerMode.WARNING: __LoggerRegistry.Instance().register_logger(__WarningLogger(stream), application_name, replace)
    elif logger_mode==LoggerMode.ERROR:   __LoggerRegistry.Instance().register_logger(__ErrorLogger(stream), application_name, replace)

def get_registered_logger_instance(application_name=None):
    return __LoggerRegistry.Instance().get_logger_instance(application_name)

def register_secondary_logger(stream=DEFAULT_STREAM, logger_mode=LoggerMode.FULL, logger_name=SECONDARY_LOGGER, application_name=None):
    if logger_mode==LoggerMode.FULL:      __SecondaryLoggerRegistry.Instance().register_logger(__FullLogger(stream), logger_name, application_name)
    elif logger_mode==LoggerMode.NONE:    __SecondaryLoggerRegistry.Instance().register_logger(__NullLogger(stream), logger_name, application_name)
    elif logger_mode==LoggerMode.WARNING: __SecondaryLoggerRegistry.Instance().register_logger(__WarningLogger(stream), logger_name, application_name)
    elif logger_mode==LoggerMode.ERROR:   __SecondaryLoggerRegistry.Instance().register_logger(__ErrorLogger(stream), logger_name, application_name)
    
def get_registered_secondary_logger(logger_name=SECONDARY_LOGGER, application_name=None):
    return __SecondaryLoggerRegistry.Instance().get_logger_instance(logger_name, application_name)
    
