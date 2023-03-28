import cv2;
import math;
import numpy as np;


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz, 3);

    indices = darkvec.argsort();
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.5
    im3 = np.empty(im.shape, im.dtype);

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz);
    return transmission


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype);

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res

def maximumBoxFilter(n, path_to_image):
  # Creates the shape of the kernel
  size = (n,n)
  shape = cv2.MORPH_RECT
  kernel = cv2.getStructuringElement(shape, size)

  # Applies the maximum filter with kernel NxN
  imgResult = cv2.dilate(path_to_image, kernel)

  # Shows the result
  cv2.namedWindow('Result with n ' + str(n), cv2.WINDOW_NORMAL) # Adjust the window length
  cv2.imshow('Result with n ' + str(n), imgResult)
  return imgResult

def TrnasmissonMap(im, A):
    for ind in range(0, 3):
        t = cv2.min(cv2.max((A[0, ind] - im[:, :, ind]) / A[0, ind], (A[0, ind] - im[:, :, ind]) / (A[0, ind]-255)), 1)
    return t

def k(src):
    dst = cv2.blur(src, (3,3))
    return dst

def i_f(src1,src2):
    alpha = 0.5
    #i_f = cv2.addWeighted(src1, alpha, src2, 1-alpha, 0, dst=None, dtype=None)\
    i_f = src1+src2

    return i_f

if __name__ == '__main__':
    import sys

    try:
        fn = sys.argv[1]
    except:
        fn = 'im6.jpg'




    def nothing(*argv):
        pass


    src = cv2.imread(fn);
    I = src.astype('float64') / 255;

    dark = DarkChannel(I, 15);
    A = AtmLight(I, dark);


    #t = maximumBoxFilter(1,te)


    tm = TrnasmissonMap(I,A);
    k = k(tm)
    TT = tm-k
    cv2.imshow("TT",TT)

    te = TransmissionEstimate(I, A, 15)
    dst = cv2.blur(te, (3, 3))
    """k1 = k(te)"""
    T1 = te-dst
    T2 = te-T1
    cv2.imshow("T1", T1)
    cv2.imshow("T2", T2)
    i_f = i_f(T2,TT)

    #J = Recover(I, t, A);
    J1 = Recover(I, i_f, A);
    cv2.imshow("dark", dark);
    cv2.imshow("t", k);
    cv2.imshow('I', src);
    cv2.imshow('Iasd', i_f);
    #cv2.imshow('J', J);
    cv2.imshow('J1', J1);
    cv2.imshow('te', te);

    cv2.imshow('tm', tm);
    cv2.waitKey();