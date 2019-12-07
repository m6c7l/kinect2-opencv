#
# Copyright (c) 2017, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

from lib.gui import *
from lib.util import *

import sys
import time
import os
import collections

# ==========================================

try:    
    import matplotlib.backends._backend_tk as mplbetk
    import matplotlib.backends.backend_agg as mplbeagg
    
    # keeping tkagg since it has become deprecated
    class FigureCanvasTkAgg(mplbeagg.FigureCanvasAgg, mplbetk.FigureCanvasTk):
    
        def draw(self):
            super(FigureCanvasTkAgg, self).draw()
            mplbetk.blit(self._tkphoto, self.renderer._renderer, (0, 1, 2, 3))
            self._master.update_idletasks()
     
        def blit(self, bbox=None):
            mplbetk.blit(self._tkphoto, self.renderer._renderer, (0, 1, 2, 3), bbox=bbox)
            self._master.update_idletasks()
                
except ImportError:
    print(e, sys.stderr)
    exit(1)

# ==========================================

try:
    import cv2 as cv
    import numpy as np
    import PIL.Image
    import PIL.ImageTk
    import PIL.ImageDraw    
except ImportError:
    print(e, sys.stderr)
    exit(1)

try:    
    import matplotlib as mpl
    import matplotlib.pyplot as mplplt
    import matplotlib.figure as mplfig
    import mpl_toolkits.mplot3d as mpl3d
    import mpl_toolkits.mplot3d.art3d as mpl3dart
except ImportError as e:
    print(e, sys.stderr)
    exit(1)

try:
    from pylibfreenect2 import Freenect2, SyncMultiFrameListener
    from pylibfreenect2 import FrameType, Registration, Frame, FrameMap
    from pylibfreenect2 import createConsoleLogger, setGlobalLogger
    from pylibfreenect2 import LoggerLevel
    try:
        from pylibfreenect2 import OpenGLPacketPipeline
        pipeline = OpenGLPacketPipeline()
    except ImportError:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
except ImportError as e:
    print(e, sys.stderr)
    exit(1)

# ==========================================

try:
    import moderngl as mgl
    import flask
except ImportError:
    pass

# ==========================================

PERSP_IR_TO_RGB = {  # projection matrices by device id
    None:   np.asarray([
                [  0.780,   0.000,  95.000],
                [  0.000,   1.155, -40.000],
                [  0.000,   0.000,   1.000]], dtype=np.float32),
    
    b'027674234847':
            np.asarray([
                [  0.774,   0.012,  91.187],
                [ -0.009,   1.146, -39.085],
                [  0.000,   0.000,   1.000]], dtype=np.float32),
    b'014110750647':
            np.asarray([
                [  0.775,  -0.008, 102.100],
                [  0.005,   1.157, -43.653],
                [  0.000,   0.000,   1.000]], dtype=np.float32),
    b'013556150647':
            np.asarray([
                [  0.783,  -0.018,  98.233],
                [  0.000,   1.163, -38.414],
                [  0.000,   0.000,   1.000]], dtype=np.float32),
    b'008736445047':
            np.asarray([
                [  0.776,  -0.003,  94.562],
                [  0.002,   1.155, -37.980],
                [  0.000,   0.000,   1.000]], dtype=np.float32),
    b'025761744747':
            np.asarray([
                [  0.777,  -0.019,  96.089],
                [  0.002,   1.146, -32-990],
                [  0.000,   0.000,   1.000]], dtype=np.float32),

}

IR_SCREEN_SIZE = (512, 424)  # pixels
    
IR_SCREEN_FOV = (70, 60)  # degrees

# ==========================================

import main

REFRESH_RATE = main.MAX_REFRESH_RATE

# ==========================================

class AppFrame(BaseFrame):

    _WIN_WITH = 1100
    _WIN_HEIGHT = 800

    def __init__(self, root=None):
        
        BaseFrame.__init__(self, root)

        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        w = self._WIN_WITH
        h = self._WIN_HEIGHT
        x = int((ws / 2) - (w / 2))
        y = int((hs / 2) - (h / 2))

        self._root = root
        self._fn = Freenect2()

        self._enabled = {True: tk.ACTIVE, False: tk.DISABLED}

        bottom = BaseFrame(self)
        bottom.pack(fill=tk.X, side=tk.BOTTOM)

        self._label = InfoLabel(bottom, text=' ', relief=tk.GROOVE, anchor=tk.W, justify=tk.LEFT, padding=(2, 2, 2, 2))
        self._label.pack(fill=tk.X, side=tk.BOTTOM)

        self._text = tk.Text(bottom, height=5)
        self._text.bind('<Key>', lambda e: 'break')
        self._text.pack(fill=tk.BOTH, side=tk.TOP)

        self._tab = TabFrame(self, tk.TOP, lambda o, n : self.switch(o, n))
        self._tab.pack(fill=tk.BOTH, expand=True)

        menu_main = tk.Menu(root)

        root.config(menu=menu_main)
        root.geometry('{}x{}+{}+{}'.format(w, h, x, y))
        root.minsize(width=400, height=300)

        menu_file = tk.Menu(menu_main, tearoff=0)
        menu_main.add_cascade(label="File", menu=menu_file)
        menu_file.add_command(label="Exit", command=self.quit)

        num_devs = self._fn.enumerateDevices()
        devs = [self._fn.getDeviceSerialNumber(i) for i in range(num_devs)]

        self._frames = []
        for srl in devs:
            self._frames.append(DeviceFrame(self._tab, self._fn, srl, lambda msg : self.show_info(msg)))
            self._tab.add(srl.decode('utf-8'), self._frames[-1])

        root.bind('<Escape>', lambda event: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.pack(fill=tk.BOTH, expand=True)

        cv.setNumThreads(1)  # since OpenCV 4.1.2
        
    def quit(self):
        self._root.destroy()
        self._root.quit()

    def show_info(self, msg, delay=2500):
        self._label.publish(msg, delay)

    def show_output(self, msg, maxlines=10):
        self._text.insert('end', msg)
        idx = int(self._text.index('end').split('.')[0]) - 2
        self._text.mark_set("insert", "{}.{}".format(idx, 0))
        self._text.see('insert')
        if idx > maxlines: self._text.delete("1.0", "2.0")
        self._text.update()

    @classmethod
    def switch(cls, odev, ndev):
        if odev is not None and ndev is not None:
            if type(odev) == DeviceFrame and type(ndev) == DeviceFrame:
                if odev != ndev:
                    odev.close()
                if not ndev.opened():
                    ndev.open()
                    ndev.play()
                elif not ndev.playing():
                    ndev.play()
                else:
                    ndev.stop()

# ==========================================

class ImageFrame(BaseFrame):

    def __init__(self, root, device, source):

        BaseFrame.__init__(self, root)

        self.master = device
        self.canvas = tk.Canvas(self)
        self.canvas.tkimg = None
        self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.grid = 0

        self.canvas.bind('<Double-Button-1>', self.__on_mouseleftbutton)
        
        self.canvas.bind('<MouseWheel>', self.__on_mousewheel)  # windows
        self.canvas.bind('<Button-4>', self.__on_mousewheel)
        self.canvas.bind('<Button-5>', self.__on_mousewheel)
        
        self.source = source
        self.refresh()

    def refresh(self):
        if self.winfo_viewable() and self.master.playing():
            img, arr, _ = self.source()
            if img is not None:
                self.canvas.img = img
                tkimg = PIL.ImageTk.PhotoImage(image=img)
                self.canvas.tkimg = tkimg
                self.canvas.config(width=tkimg.width(), height=tkimg.height())
                self.canvas.create_image(0, 0, image=tkimg, anchor=tk.NW)
                if self.grid != 0:
                    self.__draw_grid(tkimg.width() // (self.grid+1) + 1, tkimg.height() // (self.grid+1) + 1)                
        self.after(REFRESH_RATE, self.refresh)

    def capture(self, prefix='image'):
        timezone = (int(-time.timezone / 3600) + time.daylight) % 25
        tz_abbr = 'ZABCDEFGHIKLMNOPQRSTUVWXY'
        timestamp = time.strftime('%Y%m%d' + tz_abbr[timezone] + '%H%M%S')
        scriptdir = os.path.dirname(__file__)
        timedir = os.path.join(scriptdir, '../capture')
        if not os.path.exists(timedir): os.makedirs(timedir)
        ext = 'png'
        file = os.path.join(timedir, prefix + '-' + str(timestamp) + '.' + ext)
        self.canvas.img.save(file, ext)      

    def __draw_grid(self, line_dist_horz, line_dist_vert):
        h = self.canvas.winfo_height()
        w = self.canvas.winfo_width()
        for x in range(line_dist_horz, w, line_dist_horz):
            self.canvas.create_line(x, 0, x, h, fill="lightsteelblue")
        for y in range(line_dist_vert, h, line_dist_vert):
            self.canvas.create_line(0, y, w, y, fill="lightsteelblue")

    def __on_mouseleftbutton(self, event):
        self.capture()
                    
    def __on_mousewheel(self, event, secmax=11):
        def delta(event):
            if event.num == 5 or event.delta < 0: return -1 
            return 1
        self.grid = (self.grid + delta(event)) % secmax
    
# ==========================================

class TrackerFrame(BaseFrame):
    
    def __init__(self, root, device, source, publish, tracker=None):

        BaseFrame.__init__(self, root)

        self.master = device
        self.canvas = tk.Canvas(self)
        self.canvas.tkimg = None
        self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.source = source    
        self.tracker = None
        self.roi = []
                            
        self._publish = publish        

        if tracker == 'boost':  # boosting tracker

            self.canvas.bind('<Button-3>', self.__on_mouserightbutton)
            self.canvas.bind('<ButtonRelease-3>', self.__on_mouserightbutton)
            self.canvas.bind('<Motion>', self.__on_mousemotion)
            
            self.tracker = cv.TrackerBoosting_create()

        elif type(tracker) in (list, tuple, ):  # color tracker: red, green or blue things    
                    
            self.tracker = tracker
            
            bgr = np.uint8([[tracker[::-1]]])
            hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
            hsv = tuple([int(item) for item in hsv[0,0]])
            
            self.lower = (max(0, hsv[0] - 30), 100, 100)
            self.upper = (min(179, hsv[0] + 30), 255, 255)
            
            self.pts = collections.deque(maxlen=32)
        
        self.refresh()

    def refresh(self):
        
        if self.winfo_viewable() and self.master.playing():
            
            _, img, _ = self.source()
            
            if img is not None:
                
                if type(self.tracker) == cv.TrackerBoosting:  # boosting tracker

                    if len(self.roi) == 3:
                        self.tracker.clear()
                        self.tracker = cv.TrackerBoosting_create()
                        init = (self.roi[0][0], self.roi[0][1], self.roi[1][0] - self.roi[0][0], self.roi[1][1] - self.roi[0][1])
                        self.roi = []
                        ok = self.tracker.init(img, init) # initialize tracker
                        self._publish('tracker initialization {}'.format(ok))
                        
                    elif len(self.roi) == 0:
                        timer = cv.getTickCount()
                        stat, area = self.tracker.update(img)
                        fps = cv.getTickFrequency() / (cv.getTickCount() - timer);
                        if stat:
                            p1 = (int(area[0]), int(area[1]))
                            p2 = (int(area[0] + area[2]), int(area[1] + area[3]))
                            cv.rectangle(img, p1, p2, None, 2)
                            self._publish('tracking roi: {}, {}'.format(p1, p2))
                        else:
                            self._publish('tracking failure detected (perhaps no roi specified)') 

                    elif len(self.roi) > 0:
                        p1, p2 = self.roi[0], self.roi[0]
                        if len(self.roi) > 1: p2 = self.roi[1]
                        cv.rectangle(img, p1, p2, None, 2)
                        self._publish('roi: {}, {}'.format(p1, p2))
                                            
                elif type(self.tracker) in (list, tuple, ):  # color tracker
                    
                    color = self.tracker
                    
                    bgr = cv.cvtColor(img, cv.COLOR_RGBA2BGR)
                    blur = cv.GaussianBlur(bgr, (11, 11), 0)
                    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
                    mask = cv.inRange(hsv, self.lower, self.upper)
                    mask = cv.erode(mask, None, iterations=2)
                    mask = cv.dilate(mask, None, iterations=2)
                    
                    img, cof, cen, rad = encircle(img, mask, 5, color)
                    if rad is not None:
                        self.pts.appendleft(cof)
                        img, dir = trace(img, self.pts, color, 10, max(1, rad / 20.0))
                        self._publish('motion detection: {}, {}'.format(
                            {-1: 'down', 0: 'still', +1: 'up'}[dir[1]],
                            {-1: 'left', 0: 'still', +1: 'right'}[dir[0]]))            
                        
                    #img = mask

                img = PIL.Image.fromarray(img)                     
                tkimg = PIL.ImageTk.PhotoImage(image=img)
                self.canvas.tkimg = tkimg
                self.canvas.config(width=tkimg.width(), height=tkimg.height())
                self.canvas.create_image(0, 0, image=tkimg, anchor=tk.NW)
                    
        self.after(REFRESH_RATE, self.refresh)

    def __on_mouserightbutton(self, event):        
        state = event.state // 16
        if state <= 1:  # pressed
            self.roi.clear()
            self.roi.append((event.x, event.y))
            self.roi.append((event.x, event.y))
        elif state > 1:  # released
            x1, x2, y1, y2 = self.roi[0][0], self.roi[1][0], self.roi[0][1], self.roi[1][1]
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            self.roi[0] = (x1, y1)
            self.roi[1] = (x2, y2)
            self.roi.append(None)

    def __on_mousemotion(self, event):
        if len(self.roi) == 2:
            self.roi[-1] = (event.x, event.y)

# ==========================================

class PlotFrame(BaseFrame):

    _RES = 6  # less is more expensive (more points to render)
    
    def __init__(self, parent, root, serial, source, xyz=(1, 1, (0, 1)), view=(30, 30)):
        
        BaseFrame.__init__(self, parent)

        self.master = root
        
        self.source = source[0]
        self.colors = source[1]  
        
        self._pers_rgb_ir = PERSP_IR_TO_RGB[None]
        if serial in PERSP_IR_TO_RGB: self._pers_rgb_ir = PERSP_IR_TO_RGB[serial]
        
        crop = self._pers_rgb_ir[0][0]  # scaling by perspective projection
        xsize, ysize = IR_SCREEN_SIZE  # given
        xfov, yfov = IR_SCREEN_FOV  # given
        
        self.px_to_deg = (xfov / xsize + yfov / ysize) / 2 * np.sin(crop)

        self.xyz = xyz
        self.view = view        

        self.fig = mplplt.figure()
        self.ax = mpl3d.Axes3D(self.fig)    
            
        self.canvas = tk.Canvas(master=self)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=self.canvas)
        self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)           
        
        self.fig_canvas.callbacks.connect('button_press_event', self.__on_mouseleftbutton)
        
        self._interrupt = False
        self.bind("<Configure>", self.__configure)

        self.data = None
        self.colmap = None
        
        self.__init()
        self.__refresh()

    def __init(self):

        self.ax.remove()

        self.ax = mpl3d.Axes3D(self.fig)
        self.ax.view_init(azim=self.view[0], elev=self.view[1])
        
        #self.ax.set_zlabel("y [px]")
        #self.ax.set_ylabel("x [px]")
        #xw, yw = IR_SCREEN_SIZE[0] * 2, IR_SCREEN_SIZE[1] * 2

        self.ax.set_zlabel("y [mm]")
        self.ax.set_ylabel("x [mm]")
        xw = int(np.sin(np.deg2rad(IR_SCREEN_FOV[0] / 2)) * max(self.xyz[2]) * 2)
        yw = int(np.sin(np.deg2rad(IR_SCREEN_FOV[1] / 2)) * max(self.xyz[2]) * 2)
        
        self.ax.set_xlabel("z [mm]")
        self.ax.set_xlim(self.xyz[2][::-1])

        self.ax.set_ylim(((-xw // 2), (+xw // 2)))
        self.ax.set_zlim(((+yw // 2), (-yw // 2)))
        
        if self.colors is None:
            self.ax.plot(xs=[], ys=[], zs=[], marker='.', linestyle='')
        else:
            self.ax.scatter(xs=[], ys=[], zs=[], marker='.', cmap='jet')

    def __configure(self, event):
        self._interrupt = True

    def __refresh(self, draw=False, autoscale=True):
 
        r = PlotFrame._RES
        d = -75 * r + 500
        s = 0.5 * r
        if self.colors is None:
            r, s = r // 2, s / 16  # non-scatter

        if self.winfo_viewable() and self.master.playing():
 
            if not draw:
                 
                _, _, depmap = self.source()

                self.colmap = depmap // 255  # gray
                if self.colors is not None:
                    _, self.colmap, _ = self.colors()
                    self.colmap = self.colmap / 255  # rgb
                    
                a = len(depmap) // r
                b = len(depmap[0]) // r
                i = -1
                
                davg = 0
                result = []
                x, y, z, c = [None]*a*b, [None]*a*b, [None]*a*b, [None]*a*b
                for py in range(a):
                    for px in range(b):
                        i += 1
                        iy, ix = py * r, px * r
                        if depmap[iy][ix] < 0: continue  # value in valid range?
                        x[i] = ix - b * r // 2
                        y[i] = iy - a * r // 2
                        z[i] = depmap[iy][ix]
                        davg += z[i]
                        c[i] = self.colmap[iy][ix]

                        x[i] = np.tan(np.deg2rad(self.px_to_deg * x[i])) * z[i]
                        y[i] = np.tan(np.deg2rad(self.px_to_deg * y[i])) * z[i]  
                        
                        result.append((z[i], x[i], y[i], c[i]))

                if len(result) > 0:
                        
                    dmax = davg / len(z) * 2
    
                    if autoscale:                        
                        xw = int(np.sin(np.deg2rad(IR_SCREEN_FOV[0] / 2)) * dmax * 2)
                        yw = int(np.sin(np.deg2rad(IR_SCREEN_FOV[1] / 2)) * dmax * 2)
                        if xw > 2 and yw > 2:        
                            self.ax.set_ylim(((-xw // 2), (+xw // 2)))
                            self.ax.set_zlim(((+yw // 2), (-yw // 2)))
            
                    z, x, y, c = zip(*result)
                    result = sorted(zip(z, x, y, c), reverse=True)  # sort by color                                
                    self.data = list(zip(*result))  # z, x, y, c         
                
            else:
 
                if self._interrupt:
                    self._interrupt = False
                    self.after(max(REFRESH_RATE, d // 2), self.__refresh)
                    return

                if len(self.data) > 0:

                    z, x, y, c = self.data                
                    
                    #x, y, z = np.broadcast_arrays(*[np.ravel(np.ma.filled(t, np.nan)) for t in [x, y, z]])    
                    #points._offsets3d = (z, x, y)  # positions
                    #points._sizes = [s] * len(c)   # sizes set_sizes()
                    #points.set_array(np.array(c))  # colors setFacecolor(), set_edgecolor()
    
                    if self.colors is None:
                        for lines in self.ax.lines:
                            lines.remove()
                        self.ax.plot(xs=z, ys=x, zs=y, marker='.', linestyle='', c='black', markersize=s)
                    else:
                        for child in self.ax.get_children():
                            if isinstance(child, mpl3dart.Path3DCollection):
                                child.remove()
                        self.ax.scatter(xs=z, ys=x, zs=y, marker='.', cmap='jet', s=s, c=c)
                        mpl.colors._colors_full_map.cache.clear()  # avoid memory leak by clearing the cache
                    
                    self.__draw()  
            
            self.after(max(REFRESH_RATE, d), self.__refresh, not draw)
            return

        self.after(max(REFRESH_RATE, d), self.__refresh, False)

    def __draw(self):
        self.fig_canvas.draw()
        fx, fy, fw, fh = self.fig.bbox.bounds
        img = tk.PhotoImage(master=self.canvas, width=int(fw), height=int(fh))
        self.canvas.create_image(0, 0, image=img)

    def capture(self, prefix='plot'):
        timezone = (int(-time.timezone / 3600) + time.daylight) % 25
        tz_abbr = 'ZABCDEFGHIKLMNOPQRSTUVWXY'
        timestamp = time.strftime('%Y%m%d' + tz_abbr[timezone] + '%H%M%S')
        scriptdir = os.path.dirname(__file__)
        timedir = os.path.join(scriptdir, '../capture')
        if not os.path.exists(timedir): os.makedirs(timedir)
        filename = os.path.join(timedir, prefix + '-' + str(timestamp) + '.' + 'pdf')
        self.fig.savefig(filename, bbox_inches='tight', facecolor='white', edgecolor='none', transparent=True, pad_inches=0.0)
    
    def __on_mouseleftbutton(self, event):
        if event.button == 1 and event.dblclick:
            self.capture()
            
# ==========================================
    
class DeviceFrame(TabFrame):
 
    def __init__(self, master, freenect, serial, publish):

        TabFrame.__init__(self, master)
        self.pack(fill=tk.BOTH, expand=True)

        self._freenect = freenect
        self._serial = serial
        
        self._pers_rgb_ir = PERSP_IR_TO_RGB[None]
        if serial in PERSP_IR_TO_RGB: self._pers_rgb_ir = PERSP_IR_TO_RGB[serial]
            
        self._device_index = self.__device_list_index()

        self._device = None
        self._listener = None

        self._opened = False
        self._playing = False

        self.frames = FrameMap()
        self.image_buffer = {'color': (None, None, None), 'ir': (None, None, None), 'depth': (None, None, None), 'gray': (None, None, None)}

        cam = TabFrame(self, tk.TOP)
        self.add('Camera', cam)

        color = ImageFrame(cam, self, lambda : self.get_image_color())
        cam.add('RGB', color)

        cla = ImageFrame(cam, self, lambda : self.get_image_color(filters=(clahe, )))
        cam.add('CLAHE', cla)

        equ = ImageFrame(cam, self, lambda : self.get_image_color(filters=(equalize, )))
        cam.add('Equalize', equ)

        gray = ImageFrame(cam, self, lambda : self.get_image_gray())
        cam.add('Gray', gray)
         
        ir = ImageFrame(cam, self, lambda : self.get_image_ir(filters=(stretch, )))
        cam.add('IR', ir)
        ir.canvas.bind('<Motion>', self.__on_mousemotion)

        depth = ImageFrame(cam, self, lambda : self.get_image_depth(filters=()))
        cam.add('Depth', depth)
        depth.canvas.bind('<Motion>', self.__on_mousemotion)


        track = TabFrame(self, tk.TOP)
        self.add('Tracker', track)

        boos = TrackerFrame(track, self, lambda : self.get_image_color(), publish, 'boost')
        track.add('Boosting ROI', boos)

        color = TrackerFrame(track, self, lambda : self.get_image_color(), publish, (255, 0, 0))
        track.add('Color Red', color)

        colog = TrackerFrame(track, self, lambda : self.get_image_color(), publish, (0, 255, 0))
        track.add('Color Green', colog)

        colob = TrackerFrame(track, self, lambda : self.get_image_color(), publish, (0, 0, 255))
        track.add('Color Blue', colob)

        if 'matplotlib' in sys.modules:
            plot = TabFrame(self, tk.TOP)
            self.add('3D', plot)
            
            plot_pc = PlotFrame(plot, self, serial, (lambda : self.get_image_depth(), None), (960, 540, (0, 5000)))
            plot.add('Point Cloud', plot_pc)
            
            plot_dm = PlotFrame(plot, self, serial, (lambda : self.get_image_depth(), lambda : self.get_image_ir()), (960, 540, (0, 5000)))
            plot.add('Depthmap', plot_dm)
            
            plot_cm = PlotFrame(plot, self, serial, (lambda : self.get_image_depth(), lambda : self.get_image_color()), (960, 540, (0, 5000)))
            plot.add('Colormap', plot_cm)
            
            #if 'moderngl' in sys.modules:
            #    gl = TabFrame(plot, tk.TOP)
            #    plot.add(' OpenGL ', gl)

        #if 'flask' in sys.modules:
        #    stream = TabFrame(self, tk.TOP)
        #    self.add(' Stream ', stream)


        filt1 = TabFrame(self, tk.TOP)
        self.add('Depth', filt1)
        
        depthd = ImageFrame(filt1, self, lambda : self.get_image_depth(filters=(heal,)))
        filt1.add('Heal', depthd)
        depthd.canvas.bind('<Motion>', self.__on_mousemotion)

        blu = ImageFrame(filt1, self, lambda : self.get_image_depth(filters=(blur,)))
        filt1.add('Blur', blu)

        depthh = ImageFrame(filt1, self, lambda : self.get_image_depth(filters=(denoise,)))
        filt1.add('Denoise', depthh)
        depthh.canvas.bind('<Motion>', self.__on_mousemotion)

        depths = ImageFrame(filt1, self, lambda : self.get_image_depth(filters=(static,)))
        filt1.add('Static', depths)
        depths.canvas.bind('<Motion>', self.__on_mousemotion)


        filt2 = TabFrame(self, tk.TOP)
        self.add('IR', filt2)
        
        sob = ImageFrame(filt2, self, lambda : self.get_image_ir(filters=(stretch, sobel,)))
        filt2.add('Sobel', sob)

        mas = ImageFrame(filt2, self, lambda : self.get_image_ir(filters=(stretch, masking,)))
        filt2.add('Masking', mas)


        filt3 = TabFrame(self, tk.TOP)
        self.add('Features', filt3)

        can = ImageFrame(filt3, self, lambda : self.get_image_color(filters=(canny,)))
        filt3.add('Canny', can)

        hou = ImageFrame(filt3, self, lambda : self.get_image_color(filters=(hough,)))
        filt3.add('Hough', hou)

        har = ImageFrame(filt3, self, lambda : self.get_image_color(filters=(harris,)))
        filt3.add('Harris', har)

        lap = ImageFrame(filt3, self, lambda : self.get_image_color(filters=(laplacian,)))
        filt3.add('Laplacian', lap)


        filt4 = TabFrame(self, tk.TOP)
        self.add('BackSub', filt4)

        bgs1 = ImageFrame(filt4, self, lambda : self.get_image_color(filters=(bgsub,)))
        filt4.add('MOG RGB', bgs1)

        bgs2 = ImageFrame(filt4, self, lambda : self.get_image_ir(filters=(stretch, bgsub,)))
        filt4.add('MOG IR', bgs2)


        #filt5 = TabFrame(self, tk.TOP)
        #self.add('Motion', filt5)


        self._publish = publish
        self.refresh()

    def open(self):
        self._listener = SyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)
        self._device = self._freenect.openDevice(self._serial, pipeline=pipeline)
        device_index = self.__device_list_index()
        if self._device_index != device_index:  # keep track of changes in the device list
            self._device_index = device_index
            self._device.close()
            self._listener.release(self.frames)
            self.open()
            return
        self._device.setColorFrameListener(self._listener)
        self._device.setIrAndDepthFrameListener(self._listener)
        self._device.start()
        self._opened = True
        self._playing = False

    def opened(self):
        return self._opened

    def play(self):
        if not self._opened: return False
        self._playing = True
        return True

    def playing(self):
        return self._playing

    def stop(self):
        if not self._opened: return False
        self._playing = False
        return True

    def close(self):
        if not self._opened: return
        self._device.close()
        self._opened = False
        self._playing = False

    def refresh(self):
        if self._playing:
            self._listener.release(self.frames)
            for key in self.image_buffer:            
                self.image_buffer.update({key: (None, None, self.image_buffer[key][-1])})  # reset image buffer and keep depth map
            self.frames = self._listener.waitForNewFrame()
        self.after(REFRESH_RATE, self.refresh)

    def get_image_gray(self, filters=()):
        gray, _, _ = self.image_buffer['gray']
        if gray is not None: return self.image_buffer['gray']
        _, color, _ = self.get_image_color()
        gray = cv.cvtColor(color, cv.COLOR_RGBA2GRAY)
        for f in filters: gray, *_ = f(gray) 
        return self.__to_image('gray', gray)

    def get_image_color(self, filters=()):
        color, _, _ = self.image_buffer['color']
        if color is not None: return self.image_buffer['color']
        color = self.frames['color']
        color = color.asarray(dtype=np.uint8)
        color = np.flip(color, axis=(1,))
        color = cv.resize(color, (1920 // 2, 1080 // 2))
        color = cv.cvtColor(color, cv.COLOR_BGRA2RGBA)
        #color = np.rot90(color)
        #color = np.rot90(color, 3)
        for f in filters: color, *_ = f(color) 
        return self.__to_image('color', color)

    def get_image_ir(self, filters=()):
        ir, _, _ = self.image_buffer['ir']
        if ir is not None: return self.image_buffer['ir']
        ir = self.frames['ir']
        ir = ir.asarray(dtype=np.float32)
        #ir = np.rot90(ir)[5:-5]
        #ir = np.rot90(ir, 3)
        ir = np.flip(ir, axis=(1,))
        dsize = (1920 // 2, 1080 // 2)
        ir = cv.resize(ir, dsize)
        ir = cv.warpPerspective(ir, self._pers_rgb_ir, None, borderMode=cv.BORDER_CONSTANT, borderValue=65535)
        ir = ir / 65535  # normalize
        for f in filters: ir, *_ = f(ir)
        return self.__to_image('ir', ir)

    def get_image_depth(self, d_min=0, d_max=5000, filters=()):
        depth, _, _ = self.image_buffer['depth']
        if depth is not None: return self.image_buffer['depth']
        depth = self.frames['depth']
        depth = depth.asarray(dtype=np.float32)
        #depth = np.rot90(depth)[5:-5]
        #depth = np.rot90(depth, 3)
        depth = np.flip(depth, axis=(1,))
        depth = cv.resize(depth, (1920 // 2, 1080 // 2))
        depth = cv.warpPerspective(depth, self._pers_rgb_ir, None, borderMode=cv.BORDER_CONSTANT, borderValue=d_max)
        buffer = depth.astype(int)
        buffer[buffer == d_max], buffer[buffer == d_min] = -1, -1
        depth = depth / d_max  # normalize
        for f in filters: depth, *_ = f(depth)
        return self.__to_image('depth', depth, buffer)

    def __to_image(self, key, arr, arg=None):
        if arr.dtype == np.float32 and len(arr.shape) == 2:  # ir or depth, range between 0 and 1
            arr = np.asarray(arr * np.iinfo(np.uint8).max, dtype=np.uint8)
        img = PIL.Image.fromarray(arr)
        self.image_buffer[key] = (img, arr, arg)        
        return self.image_buffer[key]

    def __device_list_index(self):
        num_devs = self._freenect.enumerateDevices()
        devs = [self._freenect.getDeviceSerialNumber(i) for i in range(num_devs)]
        return devs.index(self._serial)

    def __on_mousemotion(self, event):
        msg = " x={}px y={}px ".format(event.x, event.y)        
        _, _, buffer = self.image_buffer['depth']  # _, _, buffer = self.get_image_depth()
        if buffer is not None:
            w, h = buffer.shape[::-1]
            if buffer[event.y % h][event.x % w] > 0:
                msg = msg + " d={}mm".format(buffer[event.y % h][event.x % w])
        if self._publish is not None: self._publish(msg)
        return None
