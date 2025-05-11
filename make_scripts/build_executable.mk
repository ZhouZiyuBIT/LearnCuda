# INC_DIR
# LIB_DIR
# LIBS
# CXX_SRC
# CUDA_SRC
# EXTRA_MODULE
# EXE_TARGET
# CXXFLAGS

CPPFLAGS := $(addprefix -I, $(INC_DIR))
# for gen deps makefile
CPPFLAGS += -MMD -MP
LDFLAGS := $(addprefix -L, $(LIB_DIR))
LDFLAGS += $(addprefix -l, $(LIBS))

CXX_OBJS = $(CXX_SRC:%=$(BUILD_DIR)/%.o)
CUDA_OBJS = $(CUDA_SRC:%=$(BUILD_DIR)/%.o)

DEPS := $(CXX_OBJS:.o=.d)
DEPS += $(CUDA_OBJS:.o=.d)

$(EXE_TARGET) : $(CXX_OBJS) $(CUDA_OBJS) $(EXTRA_MODULE)
	$(CXX) $(CXX_OBJS) $(CUDA_OBJS) $(EXTRA_MODULE) -o $@ $(LDFLAGS)

$(CXX_OBJS): $(BUILD_DIR)/%.o: % | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

NVCC_CXXFLAGS := $(addprefix -Xcompiler ,$(CXXFLAGS))

$(CUDA_OBJS): $(BUILD_DIR)/%.o : % | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(CPPFLAGS) $(NVCC_CXXFLAGS) -o $@ -c $<

$(BUILD_DIR) :
	mkdir -p $@

-include $(DEPS)

