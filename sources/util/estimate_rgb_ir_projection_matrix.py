#!/bin/sh
'''which' python3 > /dev/null && exec python3 "$0" "$@" || exec python "$0" "$@"
'''

#
# Copyright (c) 2017, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

# 
# naive simulated annealing optimization
# for estimation of a projection matrix
# given pairs of screen coordinates
# in RGB and IR camera space
#

import sys
import random
import numpy as np


np.warnings.filterwarnings('ignore')
np.set_printoptions(precision=4, formatter={'float': '{:0.3f}'.format})


def temperature(a, b):
    val = 0
    for i in range(0, len(a)):
        dx = a[i][0] - b[i][0] 
        dy = a[i][1] - b[i][1] 
        d = (dx**2 + dy**2)**0.5
        val += d
    return val


def cooldown(src, dst, attempts=1):
    
    temp = temperature(src, dst)
    shear, scale, shift = 0.001, 0.01, 1.0
    fail, step = 0, 0
    
    pmat = np.asarray([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]], dtype=np.float32)  # row, col
    
    pdst = tuple(np.asarray([p + (1,)]).T for p in dst)  # homogenous
    
    while temp > 0:

        step += 1
        if step > attempts: break
        
        _pmat = pmat.copy()

        a00 = (random.random() - 0.5) * scale
        a11 = (random.random() - 0.5) * scale
        a02 = (random.random() - 0.5) * shift
        a12 = (random.random() - 0.5) * shift

        a01 = (random.random() - 0.5) * shear
        a10 = (random.random() - 0.5) * shear

        _pmat += np.asarray([[a00, a01, a02],
                             [a10, a11, a12],
                             [  0,   0,   0]], dtype=np.float32)  # row, col
                               
        _pdst = tuple(np.matmul(_pmat, p) for p in pdst)
        _dst = tuple((p[0][0], p[1][0]) for p in _pdst)
        _temp = temperature(src, _dst)
        
        if _temp < temp:
            pmat = _pmat.copy()
            temp = _temp

        else:
            fail += 1

    return pmat, temp, 1 - fail / step


if __name__ == '__main__':

    # ------------------------------------------

    # 027674234847
    rgb = (449, 164), (751, 180), (721, 308), (141, 357), (148, 490), (235,   7), (752,  66),
    ir  = (457, 179), (849, 198), (810, 310), ( 59, 346), ( 73, 459), (186,  42), (850,  99),

    # 014110750647
#     rgb = (452, 170), (758, 193), (726, 324), (140, 364), (146, 495), (238,   8), (764,  69),
#     ir  = (453, 182), (849, 201), (809, 316), ( 52, 352), ( 69, 465), (176,  44), (851, 102),

    # 013556150647
#    rgb = (181, 303), (254, 13), (468, 175), (165, 496), (782, 76), (779, 193), (744, 324),
#    ir  = (107, 294), (200, 43), (474, 177), ( 96, 459), (874, 93), (874, 193), (834, 309),

    # 008736445047
#     rgb = (182, 304), (254,  14), (467, 175), (782,  79), (777, 195), (742, 325), (548, 329),
#     ir  = (114, 296), (206,  45), (479, 181), (879,  99), (880, 200), (838, 314), (586, 318),

    # 025761744747
#     rgb = (191, 315), (265,  29), (475, 187), (785, 209), (555, 343), (751, 339), (173, 508),
#     ir  = (127, 305), (222,  56), (492, 191), (891, 209), (598, 327), (851, 325), (114, 469),
    
    # ------------------------------------------
    
    proj = np.eye(3, 3, dtype=np.float32)

    for i in range(len(rgb)):  # try different subsets of coordinates (kind of outlier detection)

        print()
        print('attempt {}'.format(i+1))
        
        rgb_ = list(rgb[:])
        ir_ = list(ir[:])
        
        rgb_.pop(i)
        ir_.pop(i)

        for epoch in range(1):  # it really works just for 1 (todo)
    
            m, n = 1e5, 10
        
            avgtem, avgeff = 0, 0
            maxtem = temperature(rgb_, ir_) 
            pmat = np.zeros((3, 3), dtype=np.float32)
            
            print()
            print('annealing epoch {}\t'.format(epoch+1), end=' ')
            
            for i in range(n):
                print('.', end='', flush=True)
                mat, tem, eff = cooldown(rgb_, ir_, m)
                pmat += mat
                avgtem += tem
                avgeff += eff
        
            pmat /= n
            avgtem /= n
            avgeff /= n
            
            effic = (1 - avgtem / maxtem)
            
            print() 
            print('initial temperature\t', maxtem)
            print('current temperature\t', avgtem)
            print('efficiency         \t', avgeff * 100, '%')
            print('effectivity        \t', effic * 100, '%')
    
            if effic <= 0: break
            
            proj = pmat
    
        print()
        print('estimate of projection matrix')
        print(str(proj).strip().replace(']\n', '],\n'). replace(' [', '[').replace(' ', ', '))
