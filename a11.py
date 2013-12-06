import os, sys
from halide import *
import math

# The only Halide module  you need is halide. It includes all of Halide

def smoothGradientNormalized():
    '''use Halide to compute a 512x512 smooth gradient equal to x+y divided by 1024
    Do not worry about the schedule.
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''
    myFunc = Func()
    x, y = Var(), Var()
    myFunc[x, y] = cast(Float(32), (x + y) / 1024.0)
    output = myFunc.realize(512, 512)
    outputNP = numpy.array(Image(output))
    return (outputNP, myFunc)

def wavyRGB():
    '''Use a Halide Func to compute a wavy RGB image like that obtained by the following
    Python formula below. output[y, x, c]=(1-c)*cos(x)*cos(y)
    Do not worry about the schedule.
    Hint : you need one more domain dimension than above
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''
    myFunc = Func()
    x, y, c = Var(), Var(), Var()
    myFunc[x, y, c] = cast(Float(32), (1-c) * cos(x) * cos(y))
    output = myFunc.realize(400, 400, 3)
    outputNP = numpy.array(Image(output))
    return (outputNP, myFunc)

def luminance(im):
    '''input is assumed to be our usual numpy image representation with 3 channels.
    Use Halide to compute a 1-channel image representing 0.3R+0.6G+0.1B
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''
    input = Image(Float(32), im)
    myFunc = Func()
    x, y = Var(), Var()
    myFunc[x, y] = 0.3 * input[x, y, 0] + 0.6 * input[x, y, 1] + 0.1 * input[x, y, 2]
    output = myFunc.realize(input.width(), input.height())
    outputNP = numpy.array(Image(output))
    return (outputNP, myFunc)

def sobel(lumi):
    ''' lumi is assumed to be a 1-channel numpy array.
    Use Halide to apply a SObel filter and return the gradient magnitude.
    Return a pair (outputNP, myFunc)
    where outputNP is a numpy array and myFunc is a Halide Func'''
    input = Image(Float(32), lumi)
    x, y = Var(), Var()
    myFunc, clamped, gx, gy = Func(), Func(), Func(), Func()
    clamped[x, y] = input[clamp(x, 0, input.width()-1), clamp(y, 0, input.height()-1)]
    gx[x, y] = (- clamped[x-1, y-1] + clamped[x+1, y-1]
                - 2*clamped[x-1, y] + 2*clamped[x+1, y]
                - clamped[x-1, y+1] + clamped[x+1, y+1]) / 4.0
    gy[x, y] = (- clamped[x-1, y-1] + clamped[x-1, y+1]
                - 2*clamped[x, y-1] + 2*clamped[x, y+1]
                - clamped[x+1, y-1] + clamped[x+1, y+1]) / 4.0
    myFunc[x, y] = sqrt(gx[x, y]**2 + gy[x, y]**2)
    output = myFunc.realize(input.width(), input.height())
    outputNP = numpy.array(Image(output))
    return (outputNP, myFunc)

def pythonCodeForBoxSchedule5(lumi):
    ''' lumi is assumed to be a 1-channel numpy array.
    Write the python nested loops corresponding to the 3x3 box schedule 5
    and return a list representing the order of evaluation.
    Each time you perform a computation of blur_x or blur_y, put a triplet with the name
    of the function (string 'blur_x' or 'blur_y') and the output coordinates x and y.
    e.g. [('blur_x', 0, 0), ('blur_y', 0,0), ('blur_x', 0, 1), ...] '''
    # schedule 5:
    # blur_y.compute_root()
    # blur_x.compute_at(blur_y, x)
    L = []
    input=Image(Float(32), lumi)
    width, height = input.width()-2, input.height()-2
    for y in xrange(height):
        for x in xrange(width):
            L.append(('blur_x', x, y))
            L.append(('blur_x', x, y+1))
            L.append(('blur_x', x, y+2))
            L.append(('blur_y', x, y))
    return L

def pythonCodeForBoxSchedule6(lumi):
    ''' lumi is assumed to be a 1-channel numpy array.
    Write the python nested loops corresponding to the 3x3 box schedule 5
    and return a list representing the order of evaluation.
    Each time you perform a computation of blur_x or blur_y, put a triplet with the name
    of the function (string 'blur_x' or 'blur_y') and the output coordinates x and y.
    e.g. [('blur_x', 0, 0), ('blur_y', 0,0), ('blur_x', 0, 1), ...] '''
    # schedule 6:
    # blur_y.tile(x, y, xo, yo, xi, yi, 2, 2)
    # blur_x.compute_at(blur_y, yo)
    L = []
    input=Image(Float(32), lumi)
    width, height = input.width()-2, input.height()-2
    for yo in xrange((height+1)/2):
        # compute blur_x for each strip of y
        for yi in xrange(2+2):
            y = yo*2 + yi
            if y >= height+2: y = height+1
            for xi in xrange(width):
                L.append(('blur_x', xi, y))

        for xo in xrange((width+1)/2):
            #compute blur_y
            for yi in xrange(2):
                y = yo*2 + yi
                if y >= height: y = height-1
                for xi in xrange(2):
                    x = xo*2 + xi
                    if x >= width: x = width-1
                    L.append(('blur_y', x, y))
    return L

def pythonCodeForBoxSchedule7(lumi):
    ''' lumi is assumed to be a 1-channel numpy array.
    Write the python nested loops corresponding to the 3x3 box schedule 5
    and return a list representing the order of evaluation.
    Each time you perform a computation of blur_x or blur_y, put a triplet with the name
    of the function (string 'blur_x' or 'blur_y') and the output coordinates x and y.
    e.g. [('blur_x', 0, 0), ('blur_y', 0,0), ('blur_x', 0, 1), ...] '''
    # schedule 7
    # blur_y.split(x, xo, xi, 2)
    # blur_x.compute_at(blur_y, y)
    L = []
    input = Image(Float(32), lumi)
    width, height = input.width()-2, input.height()-2
    for y in xrange(height):
        for x in xrange(width):
            L.append(('blur_x', x, y))
            L.append(('blur_x', x, y+1))
            L.append(('blur_x', x, y+2))
        for xo in xrange((width+1)/2):
            for xi in xrange(2):
                x = 2*xo + xi
                L.append(('blur_y', x, y))
    return L

########### PART 2 ##################

def localMax(lumi):
    ''' the input is assumed to be a 1-channel image
    for each pixel, return 1.0 if it's a local maximum and 0.0 otherwise
    Don't forget to handle pixels at the boundary.
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''
    input = Image(Float(32), lumi)
    x, y = Var(), Var()
    p, isMax, myFunc = Func(), Func(), Func()
    p[x, y] = input[clamp(x, 0, input.width()-1), clamp(y, 0, input.height()-1)]
    isMax[x, y] = (p[x, y] > p[x-1, y]) & (p[x, y] > p[x+1, y]) & (p[x, y] > p[x, y-1]) & (p[x, y] > p[x, y+1])
    myFunc[x, y] = select(isMax[x, y], 1.0, 0.0)
    output = myFunc.realize(input.width(), input.height())
    outputNP = numpy.array(Image(output))
    return (outputNP, myFunc)

def GaussianSingleChannel(input, sigma, trunc=3):
    '''takes a single-channel image or Func IN HALIDE FORMAT as input
    and returns a Gaussian blurred Func with standard
    deviation sigma, truncated at trunc*sigma on both sides
    return two Funcs corresponding to the two stages blurX, blurY. This will be
    useful later for scheduling.
    We advise you use the sum() sugar
    We also advise that you first generate the kernel as a Halide Func
    You can assume that input is a clamped image and you don't need to worry about
    boundary conditions here. See calling example in test file. '''

    kernel_radius = int(sigma * trunc)
    kernel_width = 2 * kernel_radius + 1

    i, c = Var(), Var()
    kernel = Func()
    kernel[i, c] = math.e ** -((i - c)**2 / float(2 * sigma**2))

    kernelR, kSum = Var(), Var()
    kernelNormalized = Func()
    kernelR = RDom(-kernel_radius, kernel_width)
    kSum = sum(kernel[kernelR.x, 0])
    kernelNormalized[i, c] = kernel[i, c] / kSum
    kernelNormalized.compute_root()

    blurX, finalBlur = Func(), Func()
    x, y = Var(), Var()
    blurX[x, y] = sum(input[x + kernelR.x, y] * kernelNormalized[x + kernelR.x, x])
    blurX.compute_root()

    finalBlur[x, y] = sum(blurX[x, y + kernelR.x] * kernelNormalized[y + kernelR.x, y])
    return blurX, finalBlur


def harris(im, scheduleIndex):
    ''' im is a numpy RGB array.
    return the location of Harris corners like the reference Python code, but computed
    using Halide.
    when scheduleIndex is zero, just schedule all the producers of non-local consumers as root.
    when scheduleIndex is 1, use a smart schedule that makes use of parallelism and
    has decent locality (tiles are often a good option). Do not worry about vectorization.
    Note that the local maximum criterion is simplified compared to our original Harris
    You might want to reuse or copy-paste some of the code you wrote above
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''

    sigma = 5.0
    trunc = 3
    kHarris = 0.15
    thrHarris = 0.0

    inp = Image(Float(32), im)

    inpWidth = inp.width()
    inpHeight = inp.height()

    # Get luminance
    x, y = Var('x'), Var('y')
    lumi, clampedLumi = Func('lumi'), Func('clampedLumi')

    lumi[x, y] = 0.3 * inp[x, y, 0] + 0.6 * inp[x, y, 1] + 0.1 * inp[x, y, 2]
    clampedLumi[x, y] = lumi[clamp(x, 0, inpWidth-1), clamp(y, 0, inpHeight-1)]

    # Gaussian blur
    kernel_radius = int(sigma * trunc)
    kernel_width = 2 * kernel_radius + 1

    i = Var('i')
    kernel = Func('kernel')

    kernel[i] = math.e ** -(i**2 / float(2 * sigma**2))

    kernelR, kSum = Var('kernelR'), Var('kSum')
    kernelNormalized = Func('kernelNormalized')

    kernelR = RDom(-kernel_radius, kernel_width)
    kSum = sum(kernel[kernelR.x])
    kernelNormalized[i] = kernel[i] / kSum

    # Blur the luminance
    lumiBlurX, lumiBlur = Func('lumiBlurX'), Func('lumiBlur')

    lumiBlurX[x, y] = sum(clampedLumi[x + kernelR.x, y] * kernelNormalized[kernelR.x])
    lumiBlur[x, y] = sum(lumiBlurX[x, y + kernelR.x] * kernelNormalized[kernelR.x])

    # Sobel gradient
    gx, gy = Func('gx'), Func('gy')
    gx[x, y] = (- lumiBlur[x-1, y-1] + lumiBlur[x+1, y-1]
                - 2*lumiBlur[x-1, y] + 2*lumiBlur[x+1, y]
                - lumiBlur[x-1, y+1] + lumiBlur[x+1, y+1]) / 4.0
    gy[x, y] = (- lumiBlur[x-1, y-1] + lumiBlur[x-1, y+1]
                - 2*lumiBlur[x, y-1] + 2*lumiBlur[x, y+1]
                - lumiBlur[x+1, y-1] + lumiBlur[x+1, y+1]) / 4.0

    # Blur the tensor
    ix2BlurX, iy2BlurX, ixiyBlurX = Func('ix2BlurX'), Func('iy2BlurX'), Func('ixiyBlurX')

    ix2BlurX[x, y] = sum(gx[x + kernelR.x, y]**2 * kernelNormalized[kernelR.x])
    iy2BlurX[x, y] = sum(gy[x + kernelR.x, y]**2 * kernelNormalized[kernelR.x])
    ixiyBlurX[x, y] = sum(gx[x + kernelR.x, y] * gy[x + kernelR.x, y] * kernelNormalized[kernelR.x])

    ix2, iy2, ixiy = Func('ix2'), Func('iy2'), Func('ixiy')
    ix2[x, y] = sum(ix2BlurX[x, y + kernelR.x] * kernelNormalized[kernelR.x])
    iy2[x, y] = sum(iy2BlurX[x, y + kernelR.x] * kernelNormalized[kernelR.x])
    ixiy[x, y] = sum(ixiyBlurX[x, y + kernelR.x] * kernelNormalized[kernelR.x])

    # Harris response
    M = Func('M')
    M[x, y] = ((ix2[x, y] * iy2[x, y]) - ixiy[x, y]**2) - kHarris * ((ix2[x, y] + iy2[x, y])**2)

    isCorner, myFunc = Func('isCorner'), Func('myFunc')
    isCorner[x, y] = (M[x, y] > thrHarris) & (M[x, y] > M[x-1, y]) & (M[x, y] > M[x+1, y]) & (M[x, y] > M[x, y-1]) & (M[x, y] > M[x, y+1])
    myFunc[x, y] = select(isCorner[x, y], 1.0, 0.0)

    # Schedule it
    lumi.compute_root()
    clampedLumi.compute_root()
    kernel.compute_root()
    kernelNormalized.compute_root()
    lumiBlurX.compute_root()
    lumiBlur.compute_root()
    gx.compute_root()
    gy.compute_root()
    ix2BlurX.compute_root()
    iy2BlurX.compute_root()
    ixiyBlurX.compute_root()
    ix2.compute_root()
    iy2.compute_root()
    ixiy.compute_root()
    M.compute_root()
    isCorner.compute_root()

    if scheduleIndex == 1:

        tile1 = 256
        tile2 = 128
        tile3 = 64

        xo, yo, xi, yi = Var('xo'), Var('yo'), Var('xi'), Var('yi')
        lumi.compute_inline()

        myFunc.tile(x, y, xo, yo, xi, yi, tile1, tile1).parallel(yo)

        ix2.tile(x, y, xo, yo, xi, yi, tile2, tile2).parallel(yo).vectorize(xi, 8)
        iy2.tile(x, y, xo, yo, xi, yi, tile2, tile2).parallel(yo).vectorize(xi, 8)
        ixiy.tile(x, y, xo, yo, xi, yi, tile2, tile2).parallel(yo).vectorize(xi, 8)

        ix2BlurX.compute_at(ix2, xo).vectorize(x, 8).unroll(y)
        iy2BlurX.compute_at(iy2, xo).vectorize(x, 8).unroll(y)
        ixiyBlurX.compute_at(ixiy, xo).vectorize(x, 8).unroll(y)

        lumiBlur.tile(x, y, xo, yo, xi, yi, tile3, tile3).parallel(yo).vectorize(xi, 8)
        lumiBlurX.compute_at(lumiBlur, xo).vectorize(x, 8).unroll(y)


    # RUN IT
    output = myFunc.realize(inpWidth, inpHeight)
    outputNP = numpy.array(Image(output))
    return (outputNP, myFunc)


