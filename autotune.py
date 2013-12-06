import os, sys
import imageIO
from halide import *
import math

def harris(im, scheduleIndex=0, tile1X=0, tile1Y=0, tile2X=0, tile2Y=0, tile3X=0, tile3Y=0, computeAt1=None, computeAt2='xo', computeAt3='xo'):
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

    i, c = Var('i'), Var('c')
    kernel = Func('kernel')

    kernel[i, c] = math.e ** -((i - c)**2 / float(2 * sigma**2))

    kernelR, kSum = Var('kernelR'), Var('kSum')
    kernelNormalized = Func('kernelNormalized')

    kernelR = RDom(-kernel_radius, kernel_width)
    kSum = sum(kernel[kernelR.x, 0])
    kernelNormalized[i, c] = kernel[i, c] / kSum

    # Blur the luminance
    lumiBlurX, lumiBlur = Func('lumiBlurX'), Func('lumiBlur')

    lumiBlurX[x, y] = sum(clampedLumi[x + kernelR.x, y] * kernelNormalized[x + kernelR.x, x])
    lumiBlur[x, y] = sum(lumiBlurX[x, y + kernelR.x] * kernelNormalized[y + kernelR.x, y])

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

    ix2BlurX[x, y] = sum(gx[x + kernelR.x, y]**2 * kernelNormalized[x + kernelR.x, x])
    iy2BlurX[x, y] = sum(gy[x + kernelR.x, y]**2 * kernelNormalized[x + kernelR.x, x])
    ixiyBlurX[x, y] = sum(gx[x + kernelR.x, y] * gy[x + kernelR.x, y] * kernelNormalized[x + kernelR.x, x])

    ix2, iy2, ixiy = Func('ix2'), Func('iy2'), Func('ixiy')
    ix2[x, y] = sum(ix2BlurX[x, y + kernelR.x] * kernelNormalized[y + kernelR.x, y])
    iy2[x, y] = sum(iy2BlurX[x, y + kernelR.x] * kernelNormalized[y + kernelR.x, y])
    ixiy[x, y] = sum(ixiyBlurX[x, y + kernelR.x] * kernelNormalized[y + kernelR.x, y])

    # Harris response
    M = Func('M')
    M[x, y] = ((ix2[x, y] * iy2[x, y]) - ixiy[x, y]**2) - kHarris * ((ix2[x, y] + iy2[x, y])**2)

    isCorner, myFunc = Func('isCorner'), Func('myFunc')
    isCorner[x, y] = (M[x, y] > thrHarris) & (M[x, y] > M[x-1, y]) & (M[x, y] > M[x+1, y]) & (M[x, y] > M[x, y-1]) & (M[x, y] > M[x, y+1])
    myFunc[x, y] = select(isCorner[x, y], 1.0, 0.0)


    xo, yo, xi, yi = Var('xo'), Var('yo'), Var('xi'), Var('yi')

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

    if scheduleIndex != 0:
        lumi.compute_inline()
        clampedLumi.compute_root()
        kernel.compute_root()
        kernelNormalized.compute_root()

        M.compute_inline()
        isCorner.compute_inline()

        if tile1X != 0 and tile1Y != 0:
            myFunc.tile(x, y, xo, yo, xi, yi, tile1X, tile1Y).parallel(yo)

            if computeAt1 != None:
                computeAt1 = xo if computeAt1 == 'xo' else yo
                ix2.compute_at(myFunc, computeAt1)
                iy2.compute_at(myFunc, computeAt1)
                ixiy.compute_at(myFunc, computeAt1)

        if tile2X != 0 and tile2Y != 0:
            ix2.tile(x, y, xo, yo, xi, yi, tile2X, tile2Y).parallel(yo)
            iy2.tile(x, y, xo, yo, xi, yi, tile2X, tile2Y).parallel(yo)
            ixiy.tile(x, y, xo, yo, xi, yi, tile2X, tile2Y).parallel(yo)

            if computeAt2 != None:
                computeAt2 = xo if computeAt2 == 'xo' else yo
                ix2BlurX.compute_at(ix2, computeAt2)
                iy2BlurX.compute_at(iy2, computeAt2)
                ixiyBlurX.compute_at(ixiy, computeAt2)

        if tile3X != 0 and tile3Y != 0:
            lumiBlur.tile(x, y, xo, yo, xi, yi, tile3X, tile3Y).parallel(yo)

            if computeAt3 != None:
                computeAt3 = xo if computeAt3 == 'xo' else yo
                lumiBlurX.compute_at(lumiBlur, computeAt3)

    return myFunc

def harrisSameTile(im, scheduleIndex, tile1, tile2, tile3, c1, c2, c3):
    return harris(im, scheduleIndex, tile1, tile1, tile2, tile2, tile3, tile3, c1, c2, c3)

def runAndMeasure(myFunc, w, h, nTimes=1):
    L=[]
    myFunc.compile_jit()
    for i in xrange(nTimes):
        t=time.time()
        out=myFunc.realize(w,h)
        L.append (time.time()-t)
    hIm=Image(out)
    mpix=hIm.width()*hIm.height()/1e6
    print 'best: ', numpy.min(L), 'average: ', numpy.mean(L)
    print  '%.5f ms per megapixel (%.7f ms for %.2f megapixels)' % (numpy.mean(L)/mpix*1e3, numpy.mean(L)*1e3, mpix)
    return numpy.mean(L)

def testAll():
    im = imageIO.imread('rgb-small.png')
    f = harris(im)
    w, h = (im.shape[1], im.shape[0])
    print 'all root'
    best = (runAndMeasure(f, w, h), "all_root")
    computeAt = ['xo', 'yo']
    c1 = None
    for tile1 in xrange(7,11):
        for tile2 in xrange(5,tile1):
            for tile3 in xrange(5,tile1):
                for c2 in computeAt:
                    for c3 in computeAt:
                        print 2**tile1, 2**tile2, 2**tile3, c1, c2, c3
                        f = harrisSameTile(im, 1, 2**tile1, 2**tile2, 2**tile3, c1, c2, c3)
                        t = runAndMeasure(f, w, h)
                        if t < best[0]:
                            best = (t, (2**tile1, 2**tile2, 2**tile3, c1, c2, c3))
    print best

def testTileSize():
    im = imageIO.imread('hk.png')
    f = harris(im)
    w, h = (im.shape[1], im.shape[0])
    print 'all root'
    best = (runAndMeasure(f, w, h), "all_root")
    c1 = None
    c2 = 'xo'
    c3 = 'xo'
    for tile1 in xrange(7,11):
        for tile2 in xrange(5,tile1):
            for tile3 in xrange(5,tile1):
                print 2**tile1, 2**tile2, 2**tile3, c1, c2, c3
                f = harrisSameTile(im, 1, 2**tile1, 2**tile2, 2**tile3, c1, c2, c3)
                t = runAndMeasure(f, w, h)
                if t < best[0]:
                    best = (t, (2**tile1, 2**tile2, 2**tile3, c1, c2, c3))
    print best

if __name__ == '__main__':
    testTileSize()




