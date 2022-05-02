# -------------------------------------------------------------------
# makefile for module cuda examples
# -------------------------------------------------------------------
include $(PROJECT_DIR)/app/config.ezp
include $(PROJECT_DIR)/app/compiler/$(MY_SYSTEM)_$(MY_FLAVOR)_$(MY_ARCHITECTURE)_$(MY_COMPILER).mk
include $(PROJECT_DIR)/app/compiler/my_compiler.mk

FAC=$(MY_FLAVOR)/$(MY_ARCHITECTURE)/$(MY_COMPILER)
OBJECTS_DIR=$(BUILD_DIR)/$(FAC)/obj/$(MODULE)
TESTS_DIR=$(BUILD_DIR)/tests/$(FAC)/$(MODULE)
LIB_DIR=$(BUILD_DIR)/$(FAC)/lib/

CUDA_TSTS=$(shell ls *.cu)
CUDA_EXES=$(addprefix $(TESTS_DIR)/,$(subst .cu,.exe,$(CUDA_TSTS)))
CPP_INCLUDES_DIR=$(BUILD_DIR)/include

LIBRARIES=$(LIB_DIR)/lib_cume.a \
	$(LIB_DIR)/lib_essential.a

all: info create_dirs $(CUDA_EXES)

info:
	@echo "- CUFLAGS=$(CUFLAGS)"
	@echo "----------"

create_dirs:
	@mkdir -p $(OBJECTS_DIR)
	@mkdir -p $(TESTS_DIR)


$(LIBRARY): $(OBJS)
	@echo "- generate library $(LIBRARY)"
	@ar rv $@ $(OBJS) >/dev/null 2>&1
	@ranlib $@

$(TESTS_DIR)/%.exe: %.cu
	@echo "- compile $< to $@" 
	@$(NVCC) -o $@ $< $(CUFLAGS) -I$(CPP_INCLUDES_DIR) $(LIBRARIES) -lgtest -lgtest_main -lpthread

	
clean:
	rm -rf $(TESTS_DIR)/*.exe 


