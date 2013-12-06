import numpy
import imageIO
from math import *

# Tutorial 1
def rgbSmoothGradientPython():
    width=400
    height=400
    output=numpy.empty(400, 400, 3)
    for y in xrange(height):
        for x in xrange(width):
            for c in xrange(3):
                output[y, x, c]=(1-c)*cos(x)*cos(y)
    return output

# Tutorial 2
def luminance(im):
    output=numpy.empty((im.shape[0], im.shape[1], 3))
    for y in xrange(im.shape[0]):
        for x in xrange(im.shape[1]):
                output[y, x]=0.3*im[y, x, 0]+0.6*im[y, x, 1]+0.1*im[y, x, 2]
    return output

# Tutorial 4
def clamp(a, mini, maxi):
    if a<mini: a=mini
    if a>maxi: a=maxi
    return a

def pix(y, x, im):
    return im[clamp(y, 0, im.shape[0]-1),
              clamp(x, 0, im.shape[1]-1)]

def sobelMagnitude(lumi):
    '''lumi has a signle channel'''
    gx=numpy.empty(lumi.shape)
    for y in xrange(lumi.shape[0]):
        for x in xrange(lumi.shape[1]):
            gx[y,x]= (- pix(y-1, x-1, lumi) + pix(y-1, x+1, lumi)
                     - 2*pix(y, x-1, lumi) + 2*pix(y, x+1, lumi)
                     - pix(y+1, x-1, lumi) + pix(y+1, x+1, lumi) )/4.0
    gy=numpy.empty(lumi.shape)
    for y in xrange(lumi.shape[0]):
        for x in xrange(lumi.shape[1]):
            gy[y,x]= (- pix(y-1, x-1, lumi) + pix(y+1, x-1, lumi)
                     - 2*pix(y-1, x, lumi) + 2*pix(y+1, x, lumi)
                     - pix(y-1, x+1, lumi) + pix(y+1, x+1, lumi) )/4.0
    mag=numpy.empty((lumi.shape[0], lumi.shape[1], 3))
    for y in xrange(lumi.shape[0]):
        for x in xrange(lumi.shape[1]):
            mag[y,x]=sqrt(gx[y,x]**2 + gy[y,x]**2)
    return mag