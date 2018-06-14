/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
* All Rights Reserved.
*
* If this  software was obtained  under the  Intel Simplified  Software License,
* the following terms apply:
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*
*
* If this  software  was obtained  under the  Apache License,  Version  2.0 (the
* "License"), the following terms apply:
*
* You may  not use this  file except  in compliance  with  the License.  You may
* obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
*
*
* Unless  required  by   applicable  law  or  agreed  to  in  writing,  software
* distributed under the License  is distributed  on an  "AS IS"  BASIS,  WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*
* See the   License  for the   specific  language   governing   permissions  and
* limitations under the License.
*******************************************************************************/

#include <iostream>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <ippcore.h>
#include <ippi.h>
#include <ipps.h>
#include <ippcc.h>
#include <ippcv.h>
#include <mkl.h>

using namespace cv;
using namespace std;

void resize(const uchar* img_src, int src_height, int src_width, uchar* img_dst, int dst_height, int dst_width)
{
    // ippSetCpuFeatures(cpuFeatures);
    // ippSetNumThreads(numThr);

    IppiSize ssize = {src_width, src_height};
    IppiRect srect = {0, 0, src_width, src_height};
    IppiSize dsize = {dst_width, dst_height};
    IppiRect drect = {0, 0, dst_width, dst_height};

    int specSize = 0;
    int initBufSize = 0;
    ippiResizeGetSize_8u(ssize, dsize, ippLinear, 0, &specSize, &initBufSize);
    IppiResizeSpec_32f* pSpec = (IppiResizeSpec_32f*)ippsMalloc_8u(specSize);
    ippiResizeLinearInit_8u(ssize, dsize, pSpec);
    int bufSize = 0;
    ippiResizeGetBufferSize_8u(pSpec, dsize, 3, &bufSize);
    Ipp8u* pBuffer = ippsMalloc_8u(bufSize);
    IppiPoint p = {0, 0};
    ippiResizeLinear_8u_C1R((const Ipp8u*)img_src, src_width, (Ipp8u*)img_dst, dst_width, p, dsize, ippBorderRepl, 0, pSpec, pBuffer);
    ippsFree(pSpec);
    ippsFree(pBuffer);
}

void fft(const uchar* img_data, int height, int width, uchar* img_fft_data)
{
	/* Init tmp resourses */
    double *x_real = (double*)mkl_malloc(width*height*sizeof(double), 64);
    double *x_fft = (double*)mkl_malloc(width*height*sizeof(double), 64);
    MKL_Complex16 *x_out = (MKL_Complex16*)mkl_malloc((width/2+1)*height*sizeof(MKL_Complex16), 64);

	/* Configure FFT handler */
    DFTI_DESCRIPTOR_HANDLE hand = 0;
    MKL_LONG N[2];
    N[0] = height;
    N[1] = width;
    DftiCreateDescriptor(&hand, DFTI_DOUBLE, DFTI_REAL, 2, N);
    DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    MKL_LONG rs[3];
    rs[0] = 0;
    rs[1] = width;
    rs[2] = 1;
    MKL_LONG cs[3];
    cs[0] = 0;
    cs[1] = width/2+1;
    cs[2] = 1;
    DftiSetValue(hand, DFTI_INPUT_STRIDES, rs);
    DftiSetValue(hand, DFTI_OUTPUT_STRIDES, cs);
    DftiSetValue(hand, DFTI_FORWARD_SCALE, (double)1.0/(width*height));
    DftiCommitDescriptor(hand);

    /* Load image data from 8U to double array */
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            // method 1: 输入数据乘以(-1)^（i+j），即可中心化
            x_real[i*width+j] = pow(-1, i + j) * (double)img_data[i*width+j];
        }
    }

	/* Perform FFT calculation */
	DftiComputeForward(hand, x_real, x_out);

	/* Extend compressed FFT results into full matrix */
	double min = numeric_limits<double>::max();
    double max = 0;
    for (int j = 0; j < width; j++)
    {
    	for (int i = 0; i < height; i++)
        {
            MKL_Complex16 val;
            if(j < width/2+1)
            {
                val.real = x_out[i*(width/2+1)+j].real;
                val.imag = x_out[i*(width/2+1)+j].imag;
                double amp = log(sqrt(val.real*val.real+val.imag*val.imag));
                x_fft[i*width+j] = amp;
                if(amp < min)
                    min = amp;
                if(amp > max)
                    max = amp;
            }
            else
            {
                if(i == 0)
					x_fft[j] = x_fft[width-j];
                else
					x_fft[i*width+j] = x_fft[(height-i)*width+width-j];
            }
        }
    }

	/* Normalize FFT results for visualization */
	for (int i = 0; i < height*width; i++)
		img_fft_data[i] = /*img_fft_data[i]>0?255:0;*/ 255.0 * (x_fft[i] - min) / (double)(max-min);

	/* Release tmp resources */
	mkl_free(x_out);
	mkl_free(x_fft);
	mkl_free(x_real);
}

int main(int argc, char** argv)
{
	Mat img = imread("testimg.jpg", IMREAD_GRAYSCALE);
	if(img.empty())
	{
		printf("Error loading image\n");
		exit(1);
	}
	int factor = 2;
	int width_dst = img.cols / factor;
	int height_dst = img.rows / factor;
	Mat img_resize = Mat::zeros(height_dst, width_dst, CV_8UC1);
	Mat img_fft = Mat::zeros(img_resize.rows, img_resize.cols, CV_8UC1);

	resize(img.data, img.rows, img.cols, img_resize.data, img_resize.rows, img_resize.cols);
	fft(img_resize.data, img_resize.rows, img_resize.cols, img_fft.data);
	imshow("img", img);
	imshow("resize", img_resize);
	imshow("fft", img_fft);
	waitKey(0);

	return 0;
}
