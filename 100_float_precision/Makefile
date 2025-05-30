# 传入以下变量
# NVCC, GENCODE_FLAGS
# CXXFLAGS NVCCFLAGS
#
#

BUILD_DIR := $(WORKDIR)/build/100_float_precision
EXE_TARGET := float_precision_test
CXX_SRC := float_precision_main.cpp
CUDA_SRC := 

EXTRA_MODULE := $(WORKDIR)/0_utils/libutils.so

INC_DIR += $(CUDA_PATH)/include ../0_utils
LIB_DIR += $(CUDA_PATH)/lib64
LIBS := cudart cublas

include ../make_scripts/build_executable.mk

