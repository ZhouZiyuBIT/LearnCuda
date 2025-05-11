
BUILD_DIR := $(WORKDIR)/build/0_utils
SHARED_TARGET := libutils.so
CXX_SRC = device_data.cpp host_data.cpp
CUDA_SRC = random_generator_int.cu

INC_DIR += /usr/local/cuda/include .
LIB_DIR += /usr/local/cuda/lib64
LIBS = cudart


include ../make_scripts/build_shared.mk

