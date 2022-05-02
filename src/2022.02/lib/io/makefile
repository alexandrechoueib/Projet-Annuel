# -------------------------------------------------------------------
# makefile for module io
# -------------------------------------------------------------------
include $(PROJECT_DIR)/app/config.ezp
include $(PROJECT_DIR)/app/compiler/$(MY_SYSTEM)_$(MY_FLAVOR)_$(MY_ARCHITECTURE)_$(MY_COMPILER).mk
include $(PROJECT_DIR)/app/compiler/my_compiler.mk


FLARCO=$(MY_FLAVOR)_$(MY_ARCHITECTURE)_$(MY_COMPILER)
OBJECTS_DIR=$(BUILD_DIR)/$(FLARCO)/obj/$(MODULE)
LIBRARY_DIR=$(BUILD_DIR)/$(FLARCO)/lib/

CPP_SRCS=$(shell ls *.cpp)
CPP_OBJS=$(addprefix $(OBJECTS_DIR)/,$(subst .cpp,.o,$(CPP_SRCS)))
CPP_INCLUDES_DIR=$(BUILD_DIR)/include
LIBRARY=$(LIBRARY_DIR)/lib_$(MODULE).a

OBJS=$(ASM_OBJS) $(CPP_OBJS) 

all: create_dirs $(OBJS) $(LIBRARY)

info:
	@echo "- $(BUILD_DIR)"
	@echo "- $(MODULE)"
	@echo "- $(OBJECTS_DIR)"


create_dirs:
	@mkdir -p $(OBJECTS_DIR)
	@mkdir -p $(LIBRARY_DIR)

copy_includes:
	@mkdir -p $(CPP_INCLUDES_DIR)/$(MODULE)
	@cp *.h $(CPP_INCLUDES_DIR)/$(MODULE)

$(LIBRARY): $(OBJS)
	@echo "- generate library $(LIBRARY)"
	@$(AR) rv $@ $(OBJS) >/dev/null 
	@$(RANLIB) $@

$(OBJECTS_DIR)/%.o: %.cpp
	@echo "- compile " $< 
	$(CPP) -c -o $@ $< $(CFLAGS) $(OFLAGS) -I$(CPP_INCLUDES_DIR)

clean:
	rm -rf $(LIBRARY) $(OBJECTS_DIR)/*.o $(CPP_INCLUDES_DIR)/$(MODULE)/*.h


