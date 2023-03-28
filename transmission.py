import cv2;
import math;
import numpy as np;

src = cv2.imread('im6.jpg');
I = src.astype('float64') / 255;
t0 = 0.1

def DarkChannel(im, sz):
   I = im.astype('float32') / 255;  # 0~1
   b, g, r = cv2.split(I)
   dc = cv2.min(cv2.min(r, g), b);  # 가장 낮은 값 패치
   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))  # 패치 사이즈 15x15
   dark = cv2.erode(dc, kernel)  # dc를 15x15크기 커널이 지나가면서 dc에서 0 확장, 잡음제거
   return dark


def AtmLight(im, dark):
 [h, w] = im.shape[:2]
 imsz = h * w
 numpx = int(max(math.floor(imsz / 1000), 1))
 darkvec = dark.reshape(imsz)
 imvec = im.reshape(imsz, 3)
 indices = darkvec.argsort()
 indices = indices[imsz - numpx::]
 atmsum = np.zeros([1, 3])
 for ind in range(1, numpx):
     atmsum = atmsum + imvec[indices[ind]]

 A = atmsum / numpx
 return A

def J(im,A):
    for ind in range(0, 3):
        tb = cv2.min(cv2.max((A[0, ind] - im[:, :, ind]) / A[0, ind], (A[0, ind] - im[:, :, ind]) / (A[0, ind] - 255)),
                    1)
    return tb

D = DarkChannel(I,3)
A = AtmLight(I,D)
J = J(src,A)



cv2.waitKey()