
BUILD_DIR = $(WORKDIR)/build/1_print_cuda_info
EXE_TARGET = print_cuda_info
CXX_SRC = print_cuda_info.cpp
CUDA_SRC = matrix_mul.cu

INC_DIR += /usr/local/cuda/include
LIB_DIR += /usr/local/cuda/lib64
LIBS = cudart

include ../make_scripts/build_executable.mk

