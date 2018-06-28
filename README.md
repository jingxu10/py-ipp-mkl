# py-ipp-mkl Tutorial
This tutorial introduces how to take advantage of API functions of [Intel&reg; Integrated Performance Primitives (Intel&reg; IPP)](https://software.intel.com/en-us/intel-ipp) / [Intel&reg; Math Kernel Library (Intel&reg; MKL)](https://software.intel.com/en-us/mkl) in Python. In this tutorial, we will call Intel&reg; IPP API functions to resize an square-size image, then apply FFT functions of Intel&reg; MKL to yield a spectrum chart.

# Requirements
This project depends on:
* Intel&reg; IPP
* Intel&reg; MKL
* Python3
* OpenCV

# Preliminary Knowledge
### 1. ctypes, a foreign function library for Python
Both Python2 and Python3 provides an easy-to-use functionality, ctypes, to call C dynamic libraries. Utilizing this functionality is quit simple. Developers simply need to open the share library with `cdll.LoadLibrary(<shared_library_name>)`, then call C functions, that are defined in the opened dynamic library, like in C language.

Since both Intel&reg; IPP and Intel&reg; MKL provide C interface, it becomes possible to integrate them into Python applications via *ctypes*. For more detailed information about how to program with ctypes, please refer to <https://docs.python.org/3.6/library/ctypes.html>.

### 2. Make your own Intel&reg; IPP dynamic library
Intel&reg; IPP splits its funtionalities into different domains, and thus there are several dynamic libraries in the package, corresponding to each individual domain.  For this reason, functions are defined in these dynamic libraries, respectively. Since *ctypes* can only call funtions that are defined in opened dynamic libraries, it might be necessary to open multiple dynamic libraries, and you need to tell which function is defined in which dynamic library.

Fortunately, Intel&reg; IPP provides a tool, *Intel&reg; IPP Custom Library Tool*, that enables you to build your own SINGLE-FILE dynamic library that contains ALL of the Intel&reg; IPP functions that are necessary for your application. The tool is located in *<IPPROOT>/tools/custom_library_tool*. Both GUI version and CLI version are provided. For more detailed information of its usage, please refer to its *readme.htm* under the tool's directory.

# Steps
### 1. Build single-file Intel&reg; IPP dynamic library
In this project, only resize-related API functions are used. *ipp_functions.txt* contains all of these necessary Intel&reg; IPP API functions. You can select them in GUI of *Intel&reg; IPP Custom Library Tool*, and click button to generate the single-file dynamic library. Or simply pass this file to the CLI version of the tool.

Running the following command under directory of *Intel&reg; IPP Custom Library Tool* will yield the single-file dynamic library *libipp_rt.so*.

	./ipp_custom_library_tool -build -n ipp_rt -l "ipp_functions.txt" -o . -intel64 -mt -linux

### 2. Define struct type in Python
Intel&reg; IPP and Intel&reg; MKL functions sometimes have parameters that are self-defined structure types. *ctypes* provides general C compatible data types, though, we need to make these C structure types into Python ones. The following is an example.

Struct type in C:

	typedef struct {
		int x;
		int y;
	} IppiSize;

This struct type needs to be changed to the following in Python:

	class IppiSize(Structure):
		_fields_ = [("x", c_int),
			("y", c_int)]

In this project, 3 structure types of Intel&reg; IPP and 1 structure type of Intel&reg; MKL are used. They are:

	IppiSize
	IppiRect
	IppiPoint
	MKL_Complex16

Definition of these structure types can be found in header files of Intel&reg; IPP and Intel&reg; MKL.

### 2. Read image with OpenCV functions
Image I/O operations are handled by OpenCV functions, `imread/imshow`, in this project. Please compile OpenCV against Python on your develop machine, or install a pre-built OpenCV library packages using apt/yum, etc.

### 3. Call API functions of Intel&reg; IPP and/or Intel&reg; MKL
For Intel&reg; IPP, use `cdll.LoadLibrary("<Path>/libipp_rt.so")` to open the single-file Intel&reg; IPP dynamic library, and make normal function calls as in C.

For instance, in C language, we call a function as the following:

	ret = function(para1, para2, ...)

In Python with *ctypes*, we call this function as the following:

	# Suppose function(para1, para2, ...) is defined in libipp_rt.so
	ipp = cdll.LoadLibrary("<Path>/libipp_rt.so")
	ret = ipp.function(para1, para2, ...)

Please refer to function *resize* in *resize_fft.py* for more details about calling Intel&reg; IPP API functions in Python.

Calling Intel&reg; MKL API functions are much simpler, since Intel&reg; MKL already provides a single-file dynamic library, *libmkl_rt.so*, by default. It is under *<MKLROOT>/lib/<ia32|intel64>*.

### 5. Expand the FFT result into full matrix to yield the spectrum chart
FFT result of real data is a conjugate-even sequence. Due to the symmetry protperty, only part of the complex-valued sequence is stored, in Intel&reg; MKL. Thus, to get a spectrum chart, we need to expand this compressed result into a full matrix.

By default, Intel&reg; MKL stores the result in CCE format, please refer to <https://software.intel.com/en-us/articles/unpack-result-of-intel-mkl-fft-to-align-with-matlab> for details of how to unpack the CCE format data.

# Notes
1. Since Python2 also provides ctypes, it is possible to call Intel&reg; IPP / Intel&reg; MKL API functions in Python2, as well.
2. Definitions of several structure types in Intel&reg; IPP and Intel&reg; MKL, like *IppiResizeSpec_32f* (in line 71, *resize_fft.cpp*), is hidden to developers. When dealing with this kind of structure types, you can simply consider it as a general memory block. There's no need to define this kind of structure types in Python.
3. Make sure to set environment variable "LD_LIBRARY_PATH" to be paths of directories that contains all of necessary dynamic library binaries.
