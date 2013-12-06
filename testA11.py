import imageIO
import numpy
import a11
import tests

from halide import *

def main():
    im=imageIO.imread('rgb-small.png')
    lumi=im[:,:,1] #I'm lazy, I'll just use green
    smallLumi=numpy.transpose(lumi[0:6, 0:6])

    # Replace if False: by if True: once you have implement the required functions.
    # Exercises:
    if True:
        outputNP, myFunc=a11.smoothGradientNormalized()
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
        imageIO.imwrite(outputNP, 'normalizedGradient.png')

    if True:
        outputNP, myFunc=a11.wavyRGB()
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
        imageIO.imwrite(outputNP, 'rgbWave.png')

    if True:
        outputNP, myFunc = a11.luminance(im)
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
        imageIO.imwrite(outputNP, 'rgbLuminance.png')
        imageIO.imwrite(tests.luminance(im), 'rgbLuminancePython.png')

    if True:
        outputNP, myFunc=a11.sobel(lumi)
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
        imageIO.imwrite(outputNP, 'sobelMag.png')
        imageIO.imwrite(tests.sobelMagnitude(lumi), 'sobelMagPython.png')

    if False:
        L=a11.pythonCodeForBoxSchedule5(smallLumi)
        for x in L:
            print x
        print ""

    if False:
        L=a11.pythonCodeForBoxSchedule6(smallLumi)
        print "Schedule 6:"
        for x in L:
            print x
        print ""

    if False:
        L=a11.pythonCodeForBoxSchedule7(smallLumi)
        print "Schedule 7"
        for x in L:
            print x
        print ""

    if True:
        outputNP, myFunc=a11.localMax(lumi)
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
        imageIO.imwrite(outputNP, 'maxi.png')

    if False:
        testWhite = numpy.ones((50,50))
        input = Image(Float(32), testWhite)
        x, y = Var('x'), Var('y')
        clamped = Func('clamped')
        clamped[x, y] = input[clamp(x, 0, input.width()-1),
                             clamp(y, 0, input.height()-1)]
        sigma = 2.0 # IS THIS RIGHT?!?
        blurX, finalBlur= a11.GaussianSingleChannel(clamped , sigma, trunc=3)
        blurXOutput = blurX.realize(input.width(), input.height())
        blurXNP = numpy.array(Image(blurXOutput))
        imageIO.imwrite(blurXNP, 'blurXWhite.png')

    if True:
        input=Image(Float(32), lumi)
        x, y = Var('x'), Var('y')
        clamped = Func('clamped')
        clamped[x, y] = input[clamp(x, 0, input.width()-1),
                             clamp(y, 0, input.height()-1)]
        sigma = 5.0 # IS THIS RIGHT?!?
        blurX, finalBlur= a11.GaussianSingleChannel(clamped , sigma, trunc=3)

        blurXOutput = blurX.realize(input.width(), input.height())
        blurXNP = numpy.array(Image(blurXOutput))
        imageIO.imwrite(blurXNP, 'blurX.png')

        finalBlurOutput = finalBlur.realize(input.width(), input.height())
        finalBlurNP = numpy.array(Image(finalBlurOutput))
        imageIO.imwrite(finalBlurNP, 'finalBlur.png')

    if True:
        # im=numpy.load('Input/hk.npy')
        # outputNP, myFunc=a11.harris(im, 0)
        # imageIO.imwrite(outputNP, 'harris.png')
        outputNP, myFunc=a11.harris(im, 1)
        imageIO.imwrite(outputNP, 'harris_fast.png')
        print ' Dimensionality of Halide Func:', myFunc.dimensions()

    if False:
        # Timing for Harris
        im=imageIO.imread('hk.png')
        myFunc = a11.harris(im, 0)
        runAndMeasure(myFunc, im.shape[1], im.shape[0])
        myFunc = a11.harris(im, 1)
        runAndMeasure(myFunc, im.shape[1], im.shape[0])
        imageIO.imwrite(resultNP, 'harrisFast.png')

def runAndMeasure(myFunc, w, h, nTimes=5):
    L=[]
    output=None
    myFunc.compile_jit()
    for i in xrange(nTimes):
        t=time.time()
        output = myFunc.realize(w,h)
        L.append (time.time()-t)
    hIm=Image(output)
    mpix=hIm.width()*hIm.height()/1e6
    print 'best: ', numpy.min(L), 'average: ', numpy.mean(L)
    print  '%.5f ms per megapixel (%.7f ms for %.2f megapixels)' % (numpy.mean(L)/mpix*1e3, numpy.mean(L)*1e3, mpix)
    return numpy.array(hIm)


 #usual python business to declare main function in module.
if __name__ == '__main__':
    main()
