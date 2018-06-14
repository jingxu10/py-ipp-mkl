#!/usr/bin/env python3
# encoding: utf-8

""" A tutorial for using IPP/MKL C API functions in Python """

__author__ = 'Jing Xu'
__email__ = 'jing.xu@intel.com'

import math
import numpy as np
import cv2
from ctypes import *

class IppiSize(Structure):
	_fields_ = [("width", c_int),
			("height", c_int)]

class IppiRect(Structure):
	_fields_ = [("x", c_int),
			("y", c_int),
			("width", c_int),
			("height", c_int)]

class IppiPoint(Structure):
	_fields_ = [("x", c_int),
			("y", c_int)]

class MKL_Complex16(Structure):
	_fields_ = [("real", c_double),
			("imag", c_double)]

# Load dynamic libraries
ipp = cdll.LoadLibrary("./intel64/libipp_rt.so")
mkl = cdll.LoadLibrary("/opt/intel/system_studio_2018/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64/libmkl_rt.so")

def resize(img_src, img_dst):
	ssize = IppiSize(img_src.shape[1], img_src.shape[0])
	srect = IppiRect(0, 0, img_src.shape[1], img_src.shape[0])
	dsize = IppiSize(img_dst.shape[1], img_dst.shape[0])
	drect = IppiRect(0, 0, img_dst.shape[1], img_dst.shape[0])
	specSize = c_int(0)
	initBufSize = c_int(0)
	ipp.ippiResizeGetSize_8u(ssize, dsize, 2, 0, byref(specSize), byref(initBufSize))
	pSpec = ipp.ippsMalloc_8u(specSize)
	ipp.ippiResizeLinearInit_8u(ssize, dsize, pSpec)
	bufSize = c_int(0)
	ipp.ippiResizeGetBufferSize_8u(pSpec, dsize, 3, byref(bufSize))
	pBuffer = ipp.ippsMalloc_8u(bufSize)
	p = IppiPoint(0, 0)
	ipp.ippiResizeLinear_8u_C1R(img_src.ctypes.data_as(POINTER(c_ubyte)), img_src.shape[1], img_dst.ctypes.data_as(POINTER(c_ubyte)), img_dst.shape[1], p, dsize, 1, 0, c_void_p(pSpec), c_void_p(pBuffer))
	ipp.ippsFree(pSpec)
	ipp.ippsFree(pBuffer)

def fft(img_data):
	height = img_data.shape[0]
	width  = img_data.shape[1]

	cdouble_imgsize = c_double * img_data.size
	ccomplex_imgsize = MKL_Complex16 * img_data.size
	x_real = cdouble_imgsize()
	x_out  = ccomplex_imgsize()
	x_fft = np.zeros((height, width, 1), dtype=np.float64)

	# Configure FFT handler
	hand = c_void_p()
	clong2 = c_long * 2
	clong3 = c_long * 3
	N = clong2()
	N[0] = height
	N[1] = width
	mkl.DftiCreateDescriptor(byref(hand), 36, 33, 2, N)
	mkl.DftiSetValue(hand, 11, 44)
	mkl.DftiSetValue(hand, 10, 39)
	rs = clong3()
	rs[0] = 0
	rs[1] = width
	rs[2] = 1
	cs = clong3()
	cs[0] = 0
	cs[1] = int(width/2+1)
	cs[2] = 1
	mkl.DftiSetValue(hand, 12, rs)
	mkl.DftiSetValue(hand, 13, cs)
	mkl.DftiSetValue(hand, 4, c_double(1.0/img_data.size))
	mkl.DftiCommitDescriptor(hand)

	# Load image data from 8U to double array
	for i in range(0, img_data.shape[0]):
		for j in range(0, img_data.shape[1]):
			x_real[i*img_data.shape[1]+j] = c_double(math.pow(-1, i+j) * img_data[i][j])

	# Perform FFT calculation
	mkl.DftiComputeForward(hand, x_real, x_out)

	# Expand compressed FFT results into full matrix
	for j in range(0, width):
		for i in range(0, height):
			if j < width/2+1:
				val = MKL_Complex16()
				val.real = x_out[i*int(width/2+1)+j].real
				val.imag = x_out[i*int(width/2+1)+j].imag;
				x_fft[i][j] = math.log(math.sqrt(val.real*val.real+val.imag*val.imag))
			else:
				if i == 0:
					x_fft[0][j] = x_fft[0][width-j]
				else:
					x_fft[i][j] = x_fft[height-i][width-j]

	# Normalize FFT results for visualization
	fft_min = x_fft.min()
	fft_max = x_fft.max()
	x_fft = 255.0 * (x_fft - fft_min) / (fft_max - fft_min)
	return x_fft.astype(np.uint8)

img = cv2.imread("testimg.jpg", cv2.IMREAD_GRAYSCALE)
img_resize = np.zeros((int(img.shape[0]/2), int(img.shape[1]/2), 1), dtype=np.uint8)
resize(img, img_resize)
img_fft = fft(img_resize)
cv2.imshow('img', img)
cv2.imshow('img_resize', img_resize)
cv2.imshow('img_fft', img_fft)
cv2.waitKey(0)
cv2.destroyAllWindows()
