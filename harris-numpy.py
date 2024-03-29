import imageIO
import numpy as np
from scipy import ndimage
import time

Sobel=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

def harris(im, sigmaG=1, factor=4, k = 0.15, thr=0.0, debug=False):

    #compute luminance and blur
    L=np.dot(im, np.array([0.3, 0.6, 0.1]))
    L=ndimage.filters.gaussian_filter(L, sigmaG)

    #compute gradient
    gx=ndimage.convolve(L, Sobel)
    gy=ndimage.convolve(L, Sobel.T)

    #form tensor
    ix2=gx**2
    iy2=gy**2
    ixiy=gx*gy

    # blur tensor
    ix2=ndimage.filters.gaussian_filter(ix2, sigmaG*factor)
    iy2=ndimage.filters.gaussian_filter(iy2, sigmaG*factor)
    ixiy=ndimage.filters.gaussian_filter(ixiy, sigmaG*factor)

    #determinant of tensor
    det=ix2*iy2-ixiy**2
    # trace of tensor
    trace=ix2+iy2

    # Harris response
    M=det-k*trace**2


    # threshold
    thresholded=np.zeros(M.shape)
    thresholded[M>thr]=1.0


    # local maximum criterion
    # There is a more elegant way in Halide with a single predicate
    # There might be a better way in numpy as well

    notmaxi1=np.greater(M[0:-2, 1:-1], M[1:-1, 1:-1])
    notmaxi2=np.greater(M[2:, 1:-1],   M[1:-1, 1:-1])
    notmaxi3=np.greater(M[1:-1, 0:-2], M[1:-1, 1:-1])
    notmaxi4=np.greater(M[1:-1, 2:] , M[1:-1, 1:-1])
    thresholded[1:-1, 1:-1][ notmaxi1]=0.0
    thresholded[1:-1, 1:-1][ notmaxi2]=0.0
    thresholded[1:-1, 1:-1][ notmaxi3]=0.0
    thresholded[1:-1, 1:-1][ notmaxi4]=0.0

    return thresholded;


def main():

    im=imageIO.imread('hk.png')
    im = im[:im.shape[0]/1.2, :im.shape[1]/1.2]
    t=time.time()
    out=harris(im)
    dt=time.time()-t

    print 'took ', dt, 'seconds'
    mp = im.shape[0] * im.shape[1] / 1e6
    print dt / float(mp),    'seconds / mp'
    # norm=np.max(out)
    imageIO.imwrite(out)



#usual python business to declare main function in module.
if __name__ == '__main__':
    main()


