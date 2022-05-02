# -------------------------------------------------------------------
# makefile for module test trees
# -------------------------------------------------------------------
include $(PROJECT_DIR)/app/config.ezp
include $(PROJECT_DIR)/app/my_config/compiler.make
include $(PROJECT_DIR)/app/my_config/makefile.$(MY_SYSTEM)

OBJECTS_DIR=$(BUILD_DIR)/obj/$(MODULE)
TESTS_DIR=$(BUILD_DIR)/tst/$(MODULE)
LIB_DIR=$(BUILD_DIR)/lib/$(MY_DISTRIBUTION)


CPP_TSTS=$(shell ls test_dynamic_tree.cpp)
CPP_EXES=$(addprefix $(TESTS_DIR)/,$(subst .cpp,.exe,$(CPP_TSTS)))
CPP_INCLUDES_DIR=$(BUILD_DIR)/include

LIBRARIES=$(LIB_DIR)/lib_trees.a \
	$(LIB_DIR)/lib_objects.a \
	$(LIB_DIR)/lib_maths.a \
	$(LIB_DIR)/lib_essential.a


all: info create_dirs $(CPP_EXES)

info:
	@echo "- BUILD_DIR = $(BUILD_DIR)"
	@echo "- CFLAGS=$(MY_CFLAGS)"
	@echo "----------"

create_dirs:
	@mkdir -p $(OBJECTS_DIR)
	@mkdir -p $(TESTS_DIR)


$(LIBRARY): $(OBJS)
	@echo "- generate library $(LIBRARY)"
	@ar rv $@ $(OBJS) >/dev/null 2>&1
	@ranlib $@

$(TESTS_DIR)/%.exe: %.cpp
	@echo "- compile " $< 
	@$(MY_COMPILER_EXE) -o $@ $< $(MY_CFLAGS) $(MY_OFLAGS) -I$(CPP_INCLUDES_DIR) $(LIBRARIES) -lgtest -lgtest_main -lpthread

	
clean:
	rm -rf $(TESTS_DIR)/*.exe 


