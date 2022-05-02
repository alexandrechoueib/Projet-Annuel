# -------------------------------------------------------------------
# makefile for module logging
# -------------------------------------------------------------------
include $(PROJECT_DIR)/app/config.ezp
include $(PROJECT_DIR)/app/compiler/$(MY_SYSTEM)_$(MY_FLAVOR)_$(MY_ARCHITECTURE)_$(MY_COMPILER).mk
include $(PROJECT_DIR)/app/compiler/my_compiler.mk


FLARCO=$(MY_FLAVOR)_$(MY_ARCHITECTURE)_$(MY_COMPILER)
OBJECTS_DIR=$(BUILD_DIR)/$(FLARCO)/obj/
LIBRARY_DIR=$(BUILD_DIR)/$(FLARCO)/lib/

THIS_MODULE_OBJECTS_DIR=$(OBJECTS_DIR)/$(MODULE)
THIS_MODULE_INCLUDES_DIR=$(CPP_INCLUDES_DIR)/$(MODULE)
THIS_LIBRARY=$(LIBRARY_DIR)/lib_$(MODULE).a


CPP_SRCS=$(shell ls *.cpp)
CPP_OBJS=$(addprefix $(THIS_MODULE_OBJECTS_DIR)/,$(subst .cpp,.o,$(CPP_SRCS)))
CPP_INCLUDES_DIR=$(BUILD_DIR)/include


OBJS=$(ASM_OBJS) $(CPP_OBJS) 

all: create_directories info compile 

info:
	@echo "- CFLAGS = $(CFLAGS)"
	@echo "- OFLAGS = $(OFLAGS)"
	@echo "---------------------------------------"


create_directories:
	@mkdir -p $(THIS_MODULE_OBJECTS_DIR)
	@mkdir -p $(LIBRARY_DIR)


copy_include_files:
	@mkdir -p $(THIS_MODULE_INCLUDES_DIR)
	@cp *.h $(THIS_MODULE_INCLUDES_DIR)


compile: create_directories $(OBJS) $(THIS_LIBRARY)


$(THIS_LIBRARY): $(OBJS)
	@echo "- generate library $(LIBRARY)"
	@$(AR) rv $@ $(OBJS) >/dev/null 2>&1
	@$(RANLIB) $@


$(THIS_MODULE_OBJECTS_DIR)/%.o: %.cpp
	@echo "- compile " $< 
	@$(CPP) -c -o $@ $< $(CFLAGS) $(OFLAGS) -I$(CPP_INCLUDES_DIR)

$(OBJECTS_DIR)/%.o: %_nasm.asm
	@echo "- assemble " $<
	@$(NASM) -o $@ $< 
	

clean:
	rm -rf $(THIS_LIBRARY) $(THIS_MODULE_OBJECTS_DIR)/*.o $(THIS_MODULE_INCLUDES_DIR)/*.h


