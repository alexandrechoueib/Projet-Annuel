# -------------------------------------------------------------------
# makefile for module objects - tests
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

# -------------------------------------------------------------------
TESTS_DIR=$(BUILD_DIR)/$(FLARCO)/tst
TESTS_OBJ_DIR=$(TESTS_DIR)/obj
TESTS_BIN_DIR=$(TESTS_DIR)/bin
CPP_TST_SRCS=$(shell ls *.cpp)
CPP_TST_OBJS=$(addprefix $(TESTS_OBJ_DIR)/,$(subst .cpp,.o,$(CPP_TST_SRCS)))
CPP_TST_EXES=$(addprefix $(TESTS_BIN_DIR)/,$(subst .cpp,.exe,$(CPP_TST_SRCS)))

LIBRARIES_NEEDED=$(LIBRARY_DIR)/lib_essential.a \
	$(LIBRARY_DIR)/lib_objects.a
GOOGLE_TESTS_LIBS=-lgtest -lgtest_main -lpthread
# -------------------------------------------------------------------


all: create_directories info compile

info:
	@echo "- CFLAGS = $(CFLAGS)"
	@echo "- OFLAGS = $(OFLAGS)"
	@echo "---------------------------------------"


create_directories:
	@mkdir -p $(TESTS_OBJ_DIR) $(TESTS_BIN_DIR)


compile: create_directories $(CPP_TST_EXES)


$(TESTS_BIN_DIR)/%.exe: $(TESTS_OBJ_DIR)/%.o
	@echo "- generate binary $@"
	@$(CPP) -o $@ $< -I$(CPP_INCLUDES_DIR) $(LIBRARIES_NEEDED) $(GOOGLE_TESTS_LIBS)

$(TESTS_OBJ_DIR)/%.o: %.cpp
	@echo "- compile " $<
	@$(CPP) -c -o $@ $< $(CFLAGS) $(OFLAGS) -I$(CPP_INCLUDES_DIR)


clean:
	rm -rf $(TESTS_OBJ_DIR)/*.o $(TESTS_BIN_DIR)/*.exe
