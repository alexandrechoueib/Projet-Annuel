# -------------------------------------------------------------------
# makefile for examples
# -------------------------------------------------------------------
include $(PROJECT_DIR)/app/config.ezp
include $(PROJECT_DIR)/app/compiler/$(MY_SYSTEM)_$(MY_FLAVOR)_$(MY_ARCHITECTURE)_$(MY_COMPILER).mk
include $(PROJECT_DIR)/app/compiler/my_compiler.mk


FAC=$(MY_FLAVOR)/$(MY_ARCHITECTURE)/$(MY_COMPILER)
EXAMPLES_BUILD_DIR=$(BUILD_DIR)/examples/$(FAC)
LIB_DIR=$(BUILD_DIR)/$(FAC)/lib/

EXAMPLES_SRCS=prime_numbers.cpp vector_of_persons.cpp n_queens.cpp n_queens_ez.cpp
EXAMPLES_EXES=$(addprefix $(EXAMPLES_BUILD_DIR)/,$(subst .cpp,.exe,$(EXAMPLES_SRCS)))
CPP_INCLUDES_DIR=$(BUILD_DIR)/include

#LIBRARIES=$(LIB_DIR)/lib_objects_$(MY_DISTRIBUTION).a $(LIB_DIR)/lib_maths_$(MY_DISTRIBUTION).a $(LIB_DIR)/lib_essential_$(MY_DISTRIBUTION).a


all: info create_dirs $(EXAMPLES_EXES)

info:


create_dirs:
	@mkdir -p $(EXAMPLES_BUILD_DIR)


$(EXAMPLES_BUILD_DIR)/prime_numbers.exe: prime_numbers.cpp
	@echo "-  compile " $< 
	@echo "-> " $@
	@$(CPP) -o $@ $< $(CFLAGS) $(OFLAGS) -I$(CPP_INCLUDES_DIR) \
		$(LIB_DIR)/lib_objects.a $(LIB_DIR)/lib_maths.a \
		$(LIB_DIR)/lib_arguments.a $(LIB_DIR)/lib_essential.a

$(EXAMPLES_BUILD_DIR)/vector_of_persons.exe: vector_of_persons.cpp person.cpp
	@echo "-  compile " $< 
	@echo "-> " $@
	@$(CPP) -o $@ $^ $(CFLAGS) $(OFLAGS) -I$(CPP_INCLUDES_DIR) \
		$(LIB_DIR)/lib_objects.a $(LIB_DIR)/lib_maths.a \
		$(LIB_DIR)/lib_arguments.a $(LIB_DIR)/lib_essential.a

$(EXAMPLES_BUILD_DIR)/n_queens.exe: n_queens.cpp
	@echo "-  compile " $< 
	@echo "-> " $@
	@$(CPP) -o $@ $< $(CFLAGS) $(OFLAGS) -I$(CPP_INCLUDES_DIR) \
		$(LIB_DIR)/lib_logging.a \
		$(LIB_DIR)/lib_objects.a $(LIB_DIR)/lib_maths.a \
		$(LIB_DIR)/lib_arguments.a $(LIB_DIR)/lib_essential.a

$(EXAMPLES_BUILD_DIR)/n_queens_ez.exe: n_queens_ez.cpp
	@echo "-  compile " $< 
	@echo "-> " $@
	@$(CPP) -o $@ $< $(CFLAGS) $(OFLAGS) -I$(CPP_INCLUDES_DIR) \
		$(LIB_DIR)/lib_logging.a \
		$(LIB_DIR)/lib_objects.a $(LIB_DIR)/lib_maths.a \
		$(LIB_DIR)/lib_arguments.a $(LIB_DIR)/lib_essential.a


# -------------------------------------------------------------------
# Run examples
# -------------------------------------------------------------------	
run:
	@for f in $(EXAMPLES_EXES) ; do \
		echo "---------------------------------------" ;\
		echo "- Example $$f" ;\
		echo "---------------------------------------" ;\
		 $$f  ;\
	done

# -------------------------------------------------------------------
# clean
# -------------------------------------------------------------------	
clean:
	rm -rf $(EXAMPLES_DIR)/*.exe 

