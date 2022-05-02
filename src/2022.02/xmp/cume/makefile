# -------------------------------------------------------------------
# makefile for examples for cume
# -------------------------------------------------------------------
include $(PROJECT_DIR)/app/config.ezp
include $(PROJECT_DIR)/app/compiler/$(MY_SYSTEM)_$(MY_FLAVOR)_$(MY_ARCHITECTURE)_$(MY_COMPILER).mk
include $(PROJECT_DIR)/app/compiler/my_compiler.mk


FLARCO=$(MY_FLAVOR)_$(MY_ARCHITECTURE)_$(MY_COMPILER)
EXAMPLES_BUILD_DIR=$(BUILD_DIR)/$(FLARCO)/xmp/$(MODULE)
EXAMPLES_OBJ_BUILD_DIR=$(BUILD_DIR)/$(FLARCO)/xmp/$(MODULE)/obj
LIBRARY_DIR=$(BUILD_DIR)/$(FLARCO)/lib/

CPP_INCLUDES_DIR=$(BUILD_DIR)/include

# target examples
EXAMPLES_XMP_SRCS=$(shell ls *_xmp.cu)
# resources for examples
EXAMPLES_RES_SRCS=$(shell ls *.cu | grep -E -v "*_xmp.cu")


EXAMPLES_XMP_OBJS=$(addprefix $(EXAMPLES_OBJ_BUILD_DIR)/,$(subst .cu,.o,$(EXAMPLES_XMP_SRCS)))
EXAMPLES_XMP_EXES=$(addprefix $(EXAMPLES_BUILD_DIR)/,$(subst .cu,.exe,$(EXAMPLES_XMP_SRCS)))
EXAMPLES_RES_OBJS=$(addprefix $(EXAMPLES_OBJ_BUILD_DIR)/,$(subst .cu,.o,$(EXAMPLES_RES_SRCS)))
OBJS=$(EXAMPLES_RES_OBJS) $(EXAMPLES_XMP_OBJS)

.SUFFIXES: .o .cpp
.SECONDARY: $(OBJS)

OPENGL_LIBS=-lglut -lGL
LIBPNG_INSTALLED=$(shell which libpng-config)
ifneq ("$(LIBPNG_INSTALLED)","")
LIBPNG= $(shell libpng-config --libs)
CUFLAGS += -DLIBPNGPP
CPP_FLAGS += -DLIBPNGPP
endif


# -------------------------------------------------------------------
# compile examples
# -------------------------------------------------------------------
all: info create_dirs objs $(EXAMPLES_XMP_EXES)

info:
	@echo "- Generate examples for module $(MODULE)"

create_dirs:
	@mkdir -p $(EXAMPLES_BUILD_DIR) $(EXAMPLES_OBJ_BUILD_DIR)

objs: $(OBJS)
	
copy_sh:
	@for f in $(EXAMPLES_XMP_EXES) ; do \
		fb=`basename -s .exe $$f` ;\
		if test -f $$fb.sh ; then \
			cp $$fb.sh $(EXAMPLES_BUILD_DIR) ; \
		fi ; \
	done

$(EXAMPLES_OBJ_BUILD_DIR)/%.o: %.cu
	@echo "--  compile " $<
	@$(NVCC) -o $@ --compile $< $(CUFLAGS) -I$(CPP_INCLUDES_DIR)


$(EXAMPLES_BUILD_DIR)/%.exe: $(EXAMPLES_OBJ_BUILD_DIR)/%.o
	@echo "--  generate " $@
	@$(NVCC) --link -o $@ $< $(CUFLAGS) -I$(CPP_INCLUDES_DIR) \
		$(EXAMPLES_RES_OBJS)\
		$(LIBRARY_DIR)/lib_cume.a \
		$(LIBRARY_DIR)/lib_arguments.a \
		$(LIBRARY_DIR)/lib_essential.a \
		$(OPENGL_LIBS) $(LIBPNG)

# -------------------------------------------------------------------
# Run examples
# -------------------------------------------------------------------	
run_examples: copy_sh
	@for f in $(EXAMPLES_XMP_EXES) ; do \
		fb=`basename -s .exe $$f` ;\
		echo "- run example $$fb ($$f)" ;\
		echo "- run example $$fb ($$f)" >> $(OUTPUT)  ;\
		if test -f $$fb.sh ; then \
			sh $$fb.sh $$f >> $(OUTPUT) ; \
		else \
		    $$f >> $(OUTPUT) ; \
		fi ; \
	done

# -------------------------------------------------------------------
# clean
# -------------------------------------------------------------------	
clean:
	rm -rf $(EXAMPLES_BUILD_DIR) $(EXAMPLES_OBJ_BUILD_DIR)

