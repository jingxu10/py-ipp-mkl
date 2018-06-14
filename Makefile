LIBROOT=/opt/intel/system_studio_2018/compilers_and_libraries_2018.2.199/linux
IPPROOT=${LIBROOT}/ipp
MKLROOT=${LIBROOT}/mkl
COMROOT=${LIBROOT}/compiler

all: resize_fft

resize_fft: resize_fft.cpp
	@g++ resize_fft.cpp -g -std=c++11 -I${IPPROOT}/include -I${MKLROOT}/include -L./intel64 -L${MKLROOT}/lib/intel64 -L${COMROOT}/lib/intel64 -Wl,-rpath,./intel64 -Wl,-rpath,${MKLROOT}/lib/intel64 -Wl,-rpath,${COMROOT}/lib/intel64 -Wl,--no-as-needed -lipp_rt -lmkl_rt -lpthread -lm -ldl `pkg-config --cflags --libs opencv` -o resize_fft

clean:
	@rm resize_fft
