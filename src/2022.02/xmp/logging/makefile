# -------------------------------------------------------------------
# makefile for examples for logging
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
EXAMPLES_XMP_SRCS=$(shell ls *_xmp.cpp)
# resources for examples
EXAMPLES_RES_SRCS=$(shell ls *.cpp | grep -E -v "*_xmp.cpp")


EXAMPLES_XMP_OBJS=$(addprefix $(EXAMPLES_OBJ_BUILD_DIR)/,$(subst .cpp,.o,$(EXAMPLES_XMP_SRCS)))
EXAMPLES_XMP_EXES=$(addprefix $(EXAMPLES_BUILD_DIR)/,$(subst .cpp,.exe,$(EXAMPLES_XMP_SRCS)))
EXAMPLES_RES_OBJS=$(addprefix $(EXAMPLES_OBJ_BUILD_DIR)/,$(subst .cpp,.o,$(EXAMPLES_RES_SRCS)))
OBJS=$(EXAMPLES_RES_OBJS) $(EXAMPLES_XMP_OBJS)

.SUFFIXES: .o .cpp
.SECONDARY: $(OBJS)

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

$(EXAMPLES_OBJ_BUILD_DIR)/%.o: %.cpp
	@echo "--  compile " $<
	@$(CPP) -c -o $@ $< $(CFLAGS) $(OFLAGS) -I$(CPP_INCLUDES_DIR)


$(EXAMPLES_BUILD_DIR)/%.exe: $(EXAMPLES_OBJ_BUILD_DIR)/%.o
	@echo "--  generate " $@
	@$(CPP) -o $@ $< $(CFLAGS) $(OFLAGS) -I$(CPP_INCLUDES_DIR) $(EXAMPLES_RES_OBJS)\
		$(LIBRARY_DIR)/lib_logging.a \
		$(LIBRARY_DIR)/lib_essential.a

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

