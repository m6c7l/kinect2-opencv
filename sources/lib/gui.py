#
# Copyright (c) 2017, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

import tkinter as tk
import tkinter.filedialog as tkfd
import tkinter.messagebox as tkmb
import tkinter.ttk as ttk

# ==========================================

class BaseFrame(ttk.Frame):

    __widget_cfg = {'relief': tk.GROOVE, 'borderwidth': 2}

    def __init__(self, master, *args, **kwargs):
        ttk.Frame.__init__(self, master, *args, **BaseFrame.__widget_cfg, **kwargs)

# ==========================================
        
class InfoLabel(ttk.Label):

    def __init__(self, *args, **kwargs):
        ttk.Label.__init__(self, *args, **kwargs)
        self.after_task = None
        self.__animate()

    def publish(self, msg, delay=2500):
        if self.after_task is not None:
            self.after_cancel(self.after_task)
        self.config(text=msg)
        ref = self.after(delay, lambda: self.config(text=' '))
        self.after_task = ref

    def __animate(self):
        if len(self['text']) == 1:
            foo = ('\\', '|', '/', '-')
            try:
                idx = foo.index(self['text'])
            except ValueError:
                idx = -1
            self['text'] = foo[(idx+1)%len(foo)]
        self.after(500, self.__animate)

# ==========================================

class TabFrame(BaseFrame):

    __widget_cfg = {'padx': 1, 'pady': 1, 'ipadx': 4, 'ipady': 0}

    def __init__(self, master, side=tk.LEFT, switch=None):
        BaseFrame.__init__(self, master)

        self.__switch = switch

        ttk.Style().configure(type(self).__name__ + '.Toolbutton', anchor='center', padding=2, relief=tk.GROOVE)
        if ttk.Style().theme_use() == 'clam':
            ttk.Style().map(type(self).__name__ + '.Toolbutton',
                        background=[('selected', dict(ttk.Style().map('Toolbutton', 'background'))['active'])])
        self.__current_frame = None
        self.__count = 0
        self.__frame_choice = tk.IntVar(0)

        if side in (tk.TOP, tk.BOTTOM):
            self.__side = tk.LEFT
        else:
            self.__side = tk.TOP

        self.__options_frame = BaseFrame(self)
        self.__options_frame.pack(side=side, fill=tk.BOTH, expand=False, **TabFrame.__widget_cfg)

        self.pack(fill=tk.BOTH, expand=True, **TabFrame.__widget_cfg)

        self.__btns = {}
        self.__max_width = 0

    def add(self, title, fr):
        b = ttk.Radiobutton(self.__options_frame, text=title, style=type(self).__name__ + '.Toolbutton', \
                           variable=self.__frame_choice, value=self.__count, \
                           command=lambda: self.select(fr))
        b.pack(fill=tk.BOTH, side=self.__side, expand=False, **TabFrame.__widget_cfg)

        self.__btns[fr] = b

        if not self.__current_frame:
            self.__current_frame = fr
            self.select(fr)
        else:
            fr.forget()

        self.__count += 1

        if len(title) > self.__max_width: self.__max_width = len(title)
        [item.config(width=self.__max_width) for key, item in self.__btns.items()]

        return b

    def select(self, fr):
        for btn in self.__btns.values(): btn.state(['!selected'])
        if self.__switch is not None: self.__switch(self.__current_frame, fr)
        self.__current_frame.forget()
        fr.pack(fill=tk.BOTH, expand=True, **TabFrame.__widget_cfg)
        self.__btns[fr].state(['selected'])
        self.__current_frame = fr
