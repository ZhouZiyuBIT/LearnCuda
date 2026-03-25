# 传入以下变量
# NVCC, GENCODE_FLAGS
# CXXFLAGS NVCCFLAGS
#
#

BUILD_DIR := $(WORKDIR)/build/101_test_shm_bank
EXE_TARGET := test_shm_bank
CXX_SRC := test_shm_bank_main.cpp
CUDA_SRC := test_shm_bank_kernel.cu

EXTRA_MODULE := $(WORKDIR)/0_utils/libutils.so

INC_DIR += $(CUDA_PATH)/include ../0_utils
LIB_DIR += $(CUDA_PATH)/lib64
LIBS := cudart cublas

include ../make_scripts/build_executable.mk

