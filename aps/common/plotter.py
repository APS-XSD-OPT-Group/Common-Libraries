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
from aps.common.plot import gui
from aps.common.widgets.generic_widget import GenericWidget, GenericInteractiveWidget
from aps.common.widgets.context_widget import DefaultMainWindow

from PyQt5.QtCore import Qt

class PlotterFacade:
    def is_active(self): raise NotImplementedError()
    def is_saving(self): raise NotImplementedError()
    def register_context_window(self, context_key, context_window=None, use_unique_id=False): raise NotImplementedError()
    def register_save_file_prefix(self, save_file_prefix): raise NotImplementedError()
    def push_plot_on_context(self, context_key, widget_class, unique_id=None, **kwargs): raise NotImplementedError()
    def get_plots_of_context(self, context_key, unique_id=None): raise NotImplementedError()
    def get_context_container_widget(self, context_key, unique_id=None): raise  NotImplementedError()
    def get_save_file_prefix(self): raise NotImplementedError()
    def draw_context_on_widget(self, context_key, container_widget, add_context_label=True, unique_id=None, **kwargs): raise NotImplementedError()
    def draw_context(self, context_key, add_context_label=True, unique_id=None, **kwargs): raise NotImplementedError
    def show_interactive_plot(self, widget_class, container_widget, **kwargs): raise NotImplementedError()
    def show_context_window(self, context_key, unique_id=None): raise NotImplementedError()
    def raise_context_window(self, context_key, unique_id=None): raise NotImplementedError()

class PlotterMode:
    FULL         = 0
    DISPLAY_ONLY = 1
    SAVE_ONLY    = 2
    NONE         = 3
    
    @classmethod
    def get_plotter_mode(cls, plotter_mode=FULL):
        if plotter_mode==cls.FULL: return "Full" 
        if plotter_mode==cls.DISPLAY_ONLY: return "Display Only" 
        if plotter_mode==cls.SAVE_ONLY: return "Save Only" 
        if plotter_mode==cls.NONE: return "None" 

class _AbstractPlotter(PlotterFacade):

    def __init__(self, application_name=None):
        self._application_name = application_name

    @classmethod
    def _save_images(cls, plot_widget_instance, **kwargs):
        figures_to_save = plot_widget_instance.get_figures_to_save()

        if not figures_to_save is None:
            for figure_to_save in figures_to_save: figure_to_save.save_figure(**kwargs)

    @classmethod
    def _build_plot(cls, widget_class, application_name, **kwargs):
        if not issubclass(widget_class, GenericWidget): raise ValueError("Widget class is not a GenericWidget")

        try:
            plot_widget_instance = widget_class(parent=None, application_name=application_name, **kwargs)
            plot_widget_instance.build_widget(**kwargs)

            return plot_widget_instance
        except Exception as e:
            raise ValueError("Plot Widget can't be created: " + str(e))

    def register_save_file_prefix(self, save_file_prefix): self.__save_file_prefix = save_file_prefix

    def get_save_file_prefix(self): return self.__save_file_prefix

    def draw_context(self, context_key, add_context_label=True, unique_id=None, **kwargs):
        self.draw_context_on_widget(context_key, self.get_context_container_widget(context_key, unique_id), add_context_label, unique_id, **kwargs)

class _AbstractActivePlotter(_AbstractPlotter):
    def __init__(self, application_name=None):
        _AbstractPlotter.__init__(self, application_name=application_name)
        self.__plot_registry = {}
        self.__context_window_registry = {}

    def is_active(self): return True

    def _register_plot(self, context_key, plot_widget, unique_id=None):
        if not unique_id is None: context_key += "_" + unique_id

        if context_key in self.__plot_registry and not self.__plot_registry[context_key] is None:
            self.__plot_registry[context_key].append(plot_widget)
        else:
            self.__plot_registry[context_key] = [plot_widget]

    def register_context_window(self, context_key, context_window=None, use_unique_id=False):
        if context_window is None: context_window = DefaultMainWindow(context_key)
        if use_unique_id:
            unique_id = str(id(context_window))
            self.__context_window_registry[context_key + "_" + unique_id] = context_window
            return unique_id
        else:
            self.__context_window_registry[context_key] = context_window
            return None

    def get_plots_of_context(self, context_key, unique_id=None):
        if not unique_id is None: context_key += "_" + unique_id
        if context_key in self.__plot_registry: return self.__plot_registry[context_key]
        else: return None

    def get_context_container_widget(self, context_key, unique_id=None):
        if not unique_id is None: context_key += "_" + unique_id

        if context_key in self.__context_window_registry: return self.__context_window_registry[context_key].get_container_widget()
        else: return None

    def draw_context_on_widget(self, context_key, container_widget, add_context_label=True, unique_id=None, **kwargs):
        context_label = context_key
        if not unique_id is None: context_key += "_" + unique_id
        container_widget.setStyleSheet(gui.stylesheet_string)

        main_box = gui.widgetBox(container_widget, context_label if add_context_label else "", orientation="horizontal")
        main_box.layout().setAlignment(Qt.AlignCenter)
        tab_widget = gui.tabWidget(main_box)

        widths  = []
        heights = []

        if context_key in self.__plot_registry:
            plot_widget_instances = self.__plot_registry[context_key]

            for plot_widget_instance in plot_widget_instances:
                tab = gui.createTabPage(tab_widget, plot_widget_instance.get_plot_tab_name())
                tab.layout().setAlignment(Qt.AlignCenter)
                tab.layout().addWidget(plot_widget_instance)
                widths.append(plot_widget_instance.width())
                heights.append(plot_widget_instance.height())
        else:
            tab = gui.createTabPage(tab_widget, context_label)

            label = gui.widgetLabel(tab, "\n\n\n\n\n        Nothing to Display")
            label.setStyleSheet("font: 24pt")

            widths.append(500)
            heights.append(370)

        try:    tab_widget_width = kwargs["tab_widget_width"]
        except: tab_widget_width = max(widths) + 20
        try:    tab_widget_height = kwargs["tab_widget_height"]
        except: tab_widget_height = max(heights) + 35

        try:    main_box_width  = kwargs["main_box_width"]
        except: main_box_width  = tab_widget_width + 20
        try:    main_box_height = kwargs["main_box_height"]
        except: main_box_height = tab_widget_height + 40

        try:    container_widget_width  = kwargs["container_widget_width"]
        except: container_widget_width  = tab_widget_width + 25
        try:    container_widget_height = kwargs["container_widget_height"]
        except: container_widget_height = tab_widget_height + 55

        tab_widget.setFixedWidth(tab_widget_width)
        tab_widget.setFixedHeight(tab_widget_height)
        main_box.setFixedHeight(main_box_height)
        main_box.setFixedWidth(main_box_width)
        container_widget.setFixedWidth(container_widget_width)
        container_widget.setFixedHeight(container_widget_height)

        container_widget.update()

    def show_interactive_plot(self, widget_class, container_widget, **kwargs):
        if not issubclass(widget_class, GenericInteractiveWidget): raise ValueError("Widget class is not a GenericInteractiveWidget")

        try:
            interactive_widget_instance = widget_class(parent=container_widget, application_name=self._application_name, **kwargs)
            interactive_widget_instance.build_widget(**kwargs)
        except Exception as e:
            raise ValueError("Plot Widget can't be created: " + str(e))

        return widget_class.get_output(interactive_widget_instance)

    def show_context_window(self, context_key, unique_id=None):
        if not unique_id is None: context_key += "_" + unique_id
        if context_key in self.__context_window_registry: self.__context_window_registry[context_key].show()
        else: pass

    def raise_context_window(self, context_key, unique_id=None):
        if not unique_id is None: context_key += "_" + unique_id
        if context_key in self.__context_window_registry:
            window = self.__context_window_registry[context_key]
            window.setWindowFlags(window.windowFlags() | Qt.WindowStaysOnTopHint)
            window.show()
        else: pass

class FullPlotter(_AbstractActivePlotter):
    def __init__(self, application_name=None): _AbstractActivePlotter.__init__(self, application_name=application_name)
    def is_saving(self): return True
    def push_plot_on_context(self, context_key, widget_class, unique_id=None, **kwargs):
        plot_widget_instance = self._build_plot(widget_class=widget_class, application_name=self._application_name, **kwargs)
        self._register_plot(context_key, plot_widget_instance, unique_id)
        self._save_images(plot_widget_instance, **kwargs)

class DisplayOnlyPlotter(_AbstractActivePlotter):
    def __init__(self, application_name=None): _AbstractActivePlotter.__init__(self, application_name=application_name)
    def is_saving(self): return False
    def push_plot_on_context(self, context_key, widget_class, unique_id=None, **kwargs):
        self._register_plot(context_key, self._build_plot(widget_class=widget_class, application_name=self._application_name, **kwargs), unique_id)

class SaveOnlyPlotter(_AbstractActivePlotter):
    def __init__(self, application_name=None): _AbstractActivePlotter.__init__(self, application_name=application_name)
    def is_active(self): return False
    def is_saving(self): return True
    def register_context_window(self, context_key, context_window=None, use_unique_id=False): pass
    def push_plot_on_context(self, context_key, widget_class, unique_id=None, **kwargs): self._save_images(self._build_plot(widget_class=widget_class, application_name=self._application_name, **kwargs))
    def get_context_container_widget(self, context_key, unique_id=None): return None
    def get_plots_of_context(self, context_key, unique_id=None): pass
    def draw_context_on_widget(self, context_key, container_widget, add_context_label=True, unique_id=None, **kwargs): pass
    def show_interactive_plot(self, widget_class, container_widget, **kwargs): pass
    def show_context_window(self, context_key, unique_id=None): pass
    def raise_context_window(self, context_key, unique_id=None): pass

class NullPlotter(_AbstractPlotter):
    def __init__(self, application_name=None): _AbstractActivePlotter.__init__(self, application_name=application_name)
    def is_active(self): return False
    def is_saving(self): return False
    def register_context_window(self, context_key, context_window=None, use_unique_id=False): pass
    def push_plot_on_context(self, context_key, widget_class, unique_id=None, **kwargs): self._build_plot(widget_class=widget_class, application_name=self._application_name, **kwargs) # necessary for some operations
    def get_context_container_widget(self, context_key, unique_id=None): return None
    def get_plots_of_context(self, context_key, unique_id=None): pass
    def draw_context_on_widget(self, context_key, container_widget, add_context_label=True, unique_id=None, **kwargs): pass
    def show_interactive_plot(self, widget_class, container_widget, **kwargs): pass
    def show_context_window(self, context_key, unique_id=None): pass
    def raise_context_window(self, context_key, unique_id=None): pass

from aps.common.registry import GenericRegistry

@Singleton
class PlotterRegistry(GenericRegistry):

    def __init__(self):
        GenericRegistry.__init__(self, registry_name="Plotter")

    @synchronized_method
    def register_plotter(self, plotter_facade_instance, application_name=None, replace=False):
        super().register_instance(plotter_facade_instance, application_name, replace)

    @synchronized_method
    def reset(self, application_name=None):
        super().reset(application_name)

    def get_plotter_instance(self, application_name=None):
        return super().get_instance(application_name)

# -----------------------------------------------------
# Factory Methods

def register_plotter_instance(plotter_mode=PlotterMode.FULL, reset=False, application_name=None, replace=False):
    if reset: PlotterRegistry.Instance().reset()

    if plotter_mode   == PlotterMode.FULL:         PlotterRegistry.Instance().register_plotter(FullPlotter(application_name), application_name, replace)
    elif plotter_mode == PlotterMode.DISPLAY_ONLY: PlotterRegistry.Instance().register_plotter(DisplayOnlyPlotter(application_name), application_name, replace)
    elif plotter_mode == PlotterMode.SAVE_ONLY:    PlotterRegistry.Instance().register_plotter(SaveOnlyPlotter(application_name), application_name, replace)
    elif plotter_mode == PlotterMode.NONE:         PlotterRegistry.Instance().register_plotter(NullPlotter(application_name), application_name, replace)

def get_registered_plotter_instance(application_name=None):
    return PlotterRegistry.Instance().get_plotter_instance(application_name)

