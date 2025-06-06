# INC_DIR
# LIB_DIR
# LIBS
# CXX_SRC
# CUDA_SRC
# EXTRA_MODULE
# SHARED_TARGET
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

$(SHARED_TARGET) : $(CXX_OBJS) $(CUDA_OBJS) $(EXTRA_MODULE)
	$(CXX) $(CXX_OBJS) $(CUDA_OBJS) $(EXTRA_MODULE) -shared -o $@ $(LDFLAGS)

$(CXX_OBJS): $(BUILD_DIR)/%.o: % | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -c $< -o $@

NVCC_CXXFLAGS := $(addprefix -Xcompiler ,$(CXXFLAGS))
NVCC_CXXFLAGS += -Xcompiler -fPIC

$(CUDA_OBJS): $(BUILD_DIR)/%.o : % | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(CPPFLAGS) $(NVCC_CXXFLAGS) -o $@ -c $<

$(BUILD_DIR) :
	mkdir -p $@

-include $(DEPS)

