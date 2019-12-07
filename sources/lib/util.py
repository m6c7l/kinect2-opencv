#
# Copyright (c) 2017, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

import sys
import cv2 as cv
import numpy as np

try:
    import pyximport; pyximport.install()
except:
    pass

# ==========================================

def subcopy(A, B, afrom, bfrom, bto):
    afrom, bfrom, bto = map(np.asarray, [afrom, bfrom, bto])
    shape = bto - bfrom
    b = tuple(map(slice, bfrom, bto + 1))
    a = tuple(map(slice, afrom, afrom + shape + 1))
    B[b] = A[a]


def rgba2rgb(img):
    if img.dtype == np.uint8 and len(img.shape) == 3:  # convert rgba to rgb
        w, h, *_ = img.shape
        ret = np.zeros((w, h, 3), dtype=np.uint8)
        for i in range(3): ret[:,:,i] = img[:,:,i]
        img = ret    
    return img


def rgba2gray(img, conv=(0.21, 0.72, 0.07)):
    if img.dtype == np.uint8 and len(img.shape) == 3:  # convert rgba to gray
        w, h, *_ = img.shape
        ret = np.zeros((w, h), dtype=np.uint8)
        for i in range(3): ret[:,:] = np.add(ret[:,:], np.multiply(img[:,:,i], conv[i]))
        img = ret
    return img

# ==========================================

def clahe(img, threshold=1.0):
    """ Contrast Limited Adaptive Histogram Equalization (CLAHE) """
    img = rgba2rgb(img)
    b, g, r = cv.split(img)  # split on blue, green and red channels
    clahe = cv.createCLAHE(clipLimit=threshold, tileGridSize=(8, 8))
    b2 = clahe.apply(b)  # apply CLAHE to each channel
    g2 = clahe.apply(g)
    r2 = clahe.apply(r)
    return cv.merge((b2, g2, r2)),  # merge changed channels


def equalize(img):
    """ Histogram Equalization """
    img = rgba2rgb(img)
    b, g, r = cv.split(img)  # split on blue, green and red channels
    b2 = cv.equalizeHist(b)  # apply Histogram Equalization to each channel
    g2 = cv.equalizeHist(g)
    r2 = cv.equalizeHist(r)
    return cv.merge((b2, g2, r2)),  # merge equalized channels

# ==========================================

def static(img, threshold=0.01):
    call = 'static_' + sys._getframe().f_back.f_code.co_name
    h, w = img.shape
    if not call in globals():
        globals()[call] = np.zeros((h, w), dtype=img.dtype)
    prev = globals()[call]
    weights = img  
    temp = img * weights + prev * (1 - weights)
    mask = np.where(temp > prev)
    prev[mask] = temp[mask]
    globals()[call] = prev
    return prev,


def heal(img, history=5, threshold=0.05):
    call = 'heal_' + sys._getframe().f_back.f_code.co_name
    h, w = img.shape
    if not call in globals():
        globals()[call] = []
    hist = globals()[call]
    impro = np.zeros((h, w), dtype=img.dtype)    
    for i in range(len(hist)):
        mask = np.where(np.logical_and(hist[i] > np.median(hist[i]) / 2, True))
        impro[mask] = hist[i][mask]
    hist.append(img)
    if len(hist) > history:
        hist.pop(0)
    return impro,


def denoise(img, threshold=0.05):
    h, w = img.shape
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    impro = np.zeros((h, w), dtype=img.dtype)
    maxed = cv.dilate(img, kernel, iterations=1)
    idx = np.where(np.abs(maxed, img) > threshold)
    impro[idx] = maxed[idx]
    return impro,


def invert(img):
    img = 1 - img
    return img,


def stretch(img):
    avg = np.average(img[img > 0])  # avg of non-black pixels
    while abs(avg - 0.5) > 0.05:  # avg is about 5 % off mid-gray?
        d = (2 * avg) - 1
        img = img * (1 - d)
        avg = np.average(img[img > 0])
    img[img > 1] = 1
    img[img < 0] = 0
    return img,

# ==========================================    

def blur(img, size=0, cutoff=True): 
    h, w = img.shape
    if size == 0:
        k = len(np.where(img == 0.0)[0]) / ((h + w) / 2)
        if k > 0:
            s = min(int(k+1) * 3, 3)
            return blur(img, s, cutoff)    
    n = int(np.sqrt(size))
    convmat = np.ones((size, size), np.uint8)  # equal-weighted convolution matrix
    xn, yn = (size + size // 4) * n, (size + size // 8) * n    
    b = cv.dilate(img, convmat, iterations=n)
    b = cv.resize(b, (w + xn, h + yn))    
    #b = cv.blur(img, (size, size))
    #b = cv.GaussianBlur(img, (size, size), 0)
    #b = cv.medianBlur(img, min(5, size - abs(size % 2 - 1)))
    #b = cv.bilateralFilter(img, size, size * 2, size * 2)
    if cutoff: b = b[yn // 2:-yn // 2, xn // 2:-xn // 2]
    return b,

# ==========================================    

def encircle(img, mask, minsize=10, color=(127, 127, 127)):
    img = rgba2rgb(img)   
    cnts, hier = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(cnts)):
        cv.drawContours(mask, cnts, i, color, 2, cv.LINE_8, hier, 0)
    if len(cnts) > 0:       
        c = max(cnts, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(c)
        cen = (int(x), int(y))
        M = cv.moments(c)
        cof = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > minsize:
            cv.circle(img, cen, int(radius), color, 2)
            cv.circle(img, cof, 2, color, -1)    
        return img, cof, cen, radius
    return img, None, None, None


def trace(img, points, color, length=10, thick=1):
    direction = [0, 0]
    a, b = len(points), -1    
    for i in np.arange(1, len(points)):        
        if points[i-1] is None or points[i] is None: continue
        if i < a: a = i
        if i > b: b = i
        th = int(np.sqrt(points.maxlen / float(i + 1)) * thick)
        cv.line(img, points[i-1], points[i], color, th)
    if a < b:
        xn, yn = points[a]
        xo, yo = points[b]
        dx = xo - xn
        dy = yo - yn
        if np.abs(dx) > length // 2: direction[0] = -1 if np.sign(dx) == 1 else +1
        if np.abs(dy) > length // 2: direction[1] = +1 if np.sign(dy) == 1 else -1
    return img, tuple(direction)  

# ==========================================    

def sobel(img, threshold=192):
    """ (Sobel of x) and (Sobel of y) """
    img = np.asarray(img * np.iinfo(np.uint8).max, dtype=np.uint8)
    img[img > threshold] = 0
    framex = cv.Sobel(img, cv.CV_8U, 1, 0)
    datax = np.array(framex, dtype=np.uint8)
    framey = cv.Sobel(img, cv.CV_8U, 0, 1)
    datay = np.array(framey, dtype=np.uint8)
    img = np.where((datax > datay), datax, datay)
    img = np.asarray(img, dtype=np.uint8)
    return img,


def masking(img, low=2, high=253):
    """ masking by threshold (b/w) """
    img = np.asarray(img * np.iinfo(np.uint8).max, dtype=np.uint8)
    lower = np.array(low)
    upper = np.array(high)
    mask = cv.inRange(img, lower, upper)  # 1 = white, 0 = black
    mask = img * mask
    return mask, 

# ==========================================

def laplacian(img, threshold=31, peaking=(255, 0, 0)):
    """ Laplacian gradient filter """
    img, gray = rgba2rgb(img), rgba2gray(img)
    edges = cv.Laplacian(gray, cv.CV_8U)
    img[edges > threshold] = peaking
    return img, edges,


def canny(img, width=0.5, peaking=(255, 0, 0)):
    """ adaptive Canny filter, edge detector  """
    #img = np.asarray(img * np.iinfo(np.uint8).max, dtype=np.uint8)
    #img = np.asarray(img * np.iinfo(np.uint8).max, dtype=np.uint8)
    img, gray = rgba2rgb(img), rgba2gray(img)
    avg = np.average(gray)  # or median
    std = int(np.std(gray) * width)
    lower = int(max(0, avg - std))
    upper = int(min(255, avg + std))
    edges = cv.Canny(gray, lower, upper, apertureSize=3)
    img[edges == 255] = peaking
    return img, edges,


def hough(img, min_length=5, peaking=(255, 0, 0)):
    """ Hough transformation, corner detection """
    _, edges, *_ = canny(img)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, min_length)
    mask = np.zeros(img.shape[:2], np.uint8)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                p = ((x1+x2) // 2, (y1+y2) // 2)
                x, y = p
                cv.line(mask, (x, y), (x, y), 255, 2)  # dot
    img = rgba2rgb(img)
    img[mask == 255] = peaking
    return img, mask,


def harris(img, peaking=(255, 0, 0)):
    """ Harris corner detection """
    img, gray = rgba2rgb(img), rgba2gray(img)
    dest = cv.cornerHarris(src=gray, blockSize=2, ksize=5, k=0.1)
    dest = cv.dilate(dest, None)
    img[dest > 0.01 * dest.max()] = peaking
    return img, dest,

# ==========================================

def bgsub(img):
    """ Background subtraction (i.e. motion detection) with given algorithm (e.g. Gaussian-Mixture model) """
    call = 'fgbg_' + sys._getframe().f_back.f_code.co_name
    if not call in globals():
        globals()[call] = cv.bgsegm.createBackgroundSubtractorMOG(history=20, nmixtures=10, backgroundRatio=0.75, noiseSigma=0.0)  # declare a global    
    fgbg = globals()[call]
    img = rgba2rgb(img)    
    img = np.asarray(img * np.iinfo(np.uint8).max, dtype=np.uint8)
    mask = fgbg.apply(img) 
    return mask,


# def diff(img, history=5, threshold=0.05):
#     call = 'heal_' + sys._getframe().f_back.f_code.co_name
#     h, w = img.shape
#     if not call in globals():
#         globals()[call] = []
#     hist = globals()[call]
#     impro = np.zeros((h, w), dtype=img.dtype)    
#     for i in range(len(hist)):
#         mask = np.where(np.logical_and(hist[i] > np.median(hist[i]) / 2, True))
#         impro[mask] = hist[i][mask]
#     hist.append(img)
#     if len(hist) > history:
#         hist.pop(0)
#     return impro,


# def filter_sift(self):
#     """ Scale-Invariant Feature Transform (SIFT). It is patented and not totally free """
#     try:
#         return self.get_features(cv2.xfeatures2d.SIFT_create())
#     except cv2.error:
#         return self.frame  # return unchanged frame


# def filter_surf(self):
#     """ Speeded-Up Robust Features (SURF). It is patented and not totally free """
#     try:
#         return self.get_features(cv2.xfeatures2d.SURF_create(4000))
#     except cv2.error:
#         return self.frame  # return unchanged frame


# def filter_orb(self):
#     """ Oriented FAST and Rotated BRIEF (ORB). It is not patented and totally free """
#     return self.get_features(cv2.ORB_create())


# def filter_brief(self):
#     """ BRIEF descriptors with the help of CenSurE (STAR) detector """
#     gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # convert to gray scale
#     keypoints = cv2.xfeatures2d.StarDetector_create().detect(gray, None)
#     keypoints, descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create().compute(gray, keypoints)
#     return cv2.drawKeypoints(image=self.frame, outImage=self.frame, keypoints=keypoints,
#                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(51, 163, 236))
        
        
# def filter_optflow(self):
#     """ Lucas Kanade optical flow """
#     gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
#     frame = self.frame.copy()  # copy the frame
#     if self.previous is None or self.previous.shape != gray.shape:
#         self.previous = gray.copy()  # save previous gray frame
#         # Find new corner points of the frame
#         self.opt_flow['points'] = cv2.goodFeaturesToTrack(
#             gray, mask=None,
#             **self.opt_flow['feature_params'])
#         # Create a new mask image for drawing purposes
#         self.opt_flow['mask'] = np.zeros_like(self.frame.copy())
#     # If motion is large this method will fail. Ignore exceptions
#     try:
#         # Calculate optical flow. cv2.error could happen here.
#         points, st, err = cv2.calcOpticalFlowPyrLK(
#             self.previous, gray,
#             self.opt_flow['points'], None, **self.opt_flow['lk_params'])
#         # Select good points
#         good_new = points[st == 1]  # TypeError 'NoneType' could happen here
#         good_old = self.opt_flow['points'][st == 1]
#         # Draw the tracks
#         for i, (new, old) in enumerate(zip(good_new, good_old)):
#             a, b = new.ravel()
#             c, d = old.ravel()
#             # Draw lines in the mask
#             self.opt_flow['mask'] = cv2.line(self.opt_flow['mask'], (a, b), (c, d),
#                                              self.opt_flow['color'][i].tolist(), 2)
#             # Draw circles in the frame
#             frame = cv2.circle(frame, (a, b), 5, self.opt_flow['color'][i].tolist(), -1)
#         # Update the previous frame and previous points
#         self.previous = gray.copy()
#         self.opt_flow['points'] = good_new.reshape(-1, 1, 2)
#         return cv2.add(frame, self.opt_flow['mask'])  # concatenate frame and mask images
#     except (TypeError, cv2.error):
#         self.previous = None  # set optical flow to None if exception occurred
#         return self.frame  # return unchanged frame when error


# def filter_motion(self):
#     """ Motion detection """
#     if self.previous is None or self.previous.shape != self.frame.shape:
#         self.previous = self.frame.copy()  # remember previous frame
#         return self.frame  # return unchanged frame
#     gray1 = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
#     gray2 = cv2.cvtColor(self.previous, cv2.COLOR_BGR2GRAY)
#     self.previous = self.frame.copy()  # remember previous frame
#     return cv2.absdiff(gray1, gray2)  # get absolute difference between two frames


# def filter_threshold(self):
#     """ Adaptive Gaussian threshold """
#     gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # convert to gray scale
#     return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# ==========================================

import os, threading


class PipeCapture:

    ESC = b'\x1b'  # 27

    def __init__(self, stream, isthread=True):
        self._stream = stream
        self._isthread = isthread
        self._descriptor = self._stream.fileno()
        self._pipe_out, self._pipe_in = os.pipe()
        self._worker = None
        self._descriptor_dub = None
        self._publish = None
        self._buffer = []

    def open(self, publish):
        self._publish = publish
        self._descriptor_dub = os.dup(self._descriptor)
        os.dup2(self._pipe_in, self._descriptor)
        if self._isthread:
            self._worker = threading.Thread(target=self.read)
            self._worker.start()

    def close(self):
        if self._publish is None: return
        self._publish = None
        self._stream.write(PipeCapture.ESC.decode('utf-8'))
        self._stream.flush()
        if self._isthread:
            self._worker.join()
        os.close(self._pipe_out)
        os.dup2(self._descriptor_dub, self._descriptor)

    def read(self):
        while self._publish is not None:
            char = os.read(self._pipe_out, 1)
            if char == PipeCapture.ESC: break
            self._buffer.append(char.decode('utf-8'))
            if self._buffer[-1] == '\n':            
                self._publish(''.join(self._buffer))
                self._buffer.clear()
