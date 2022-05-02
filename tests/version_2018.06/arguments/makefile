# -------------------------------------------------------------------
# makefile for module test essential
# -------------------------------------------------------------------
include $(PROJECT_DIR)/app/config.ezp
include $(PROJECT_DIR)/app/compiler/$(MY_SYSTEM)_$(MY_FLAVOR)_$(MY_ARCHITECTURE)_$(MY_COMPILER).mk
include $(PROJECT_DIR)/app/compiler/my_compiler.mk

FAC=$(MY_FLAVOR)/$(MY_ARCHITECTURE)/$(MY_COMPILER)
OBJECTS_DIR=$(BUILD_DIR)/$(FAC)/obj/$(MODULE)
TESTS_DIR=$(BUILD_DIR)/tests/$(FAC)/$(MODULE)
LIB_DIR=$(BUILD_DIR)/$(FAC)/lib/

CPP_TSTS=$(shell ls *.cpp)
CPP_EXES=$(addprefix $(TESTS_DIR)/,$(subst .cpp,.exe,$(CPP_TSTS)))
CPP_INCLUDES_DIR=$(BUILD_DIR)/include

LIBRARIES=$(LIB_DIR)/lib_arguments.a \
	$(LIB_DIR)/lib_essential.a

all: info create_dirs $(CPP_EXES)

info:
	@echo "- BUILD_DIR = $(BUILD_DIR)"
	@echo "- CFLAGS=$(CFLAGS)"
	@echo "- OFLAGS=$(OFLAGS)"
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
	@$(CPP) -o $@ $< $(CFLAGS) $(OFLAGS) -I$(CPP_INCLUDES_DIR) $(LIBRARIES) -lgtest -lgtest_main -lpthread

	
clean:
	rm -rf $(TESTS_DIR)/*.exe 


