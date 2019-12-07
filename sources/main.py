#!/bin/sh
'''which' python3 > /dev/null && exec python3 "$0" "$@" || exec python "$0" "$@"
'''

#
# Copyright (c) 2017, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

# 
# Kinect for Windows v2 / Kinect for Xbox 360
# Depth range:     from 0.5m to 4.5m
# Color stream:    1920×1080
# Depth stream:    512×424
# Infrared stream: 512×424
# Field of view:   h=70°, v=60°
# Audio stream:    4-mic array
# USB:             3.0
#

from lib.util import *
from lib.app import *


MAX_REFRESH_RATE = 100  # ms, applies for all visualizations


if __name__ == '__main__':

#     import tracemalloc
#     tracemalloc.start()

    root = tk.Tk()

    general_font = (None, 10, 'normal')
    root.option_add('*Font', general_font)

    root.title('Kinect2 along with OpenCV featuring Tk/Tcl in Action')

    out = PipeCapture(sys.stdout)
    app = AppFrame(root)
    out.open(lambda msg : app.show_output(msg))

    root.mainloop()
    out.close()
    
#     snapshot = tracemalloc.take_snapshot()
#     top_stats = snapshot.statistics('lineno')
#     for stat in top_stats[:10]: print(stat)
