# 传入以下变量
# NVCC, GENCODE_FLAGS
# CXXFLAGS NVCCFLAGS
#
#

BUILD_DIR := $(WORKDIR)/build/3_vector_reduction
EXE_TARGET := vector_reduction
CXX_SRC := vector_reduction_main.cpp
CUDA_SRC := reduce.cu

EXTRA_MODULE := $(WORKDIR)/0_utils/libutils.so

INC_DIR += $(CUDA_PATH)/include ../0_utils
LIB_DIR += $(CUDA_PATH)/lib64
LIBS := cudart cublas

include ../make_scripts/build_executable.mk

