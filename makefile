# ===================================================================
# makefile for project EZLIB
# ===================================================================
include app/config.ezp

# -------------------------------------------------------------------
# Modules to compile, run as examples or test
# We need to follow the dependency order, i.e, essential must be
# compiled first, then arguments which is based on essential and
# so on.
# -------------------------------------------------------------------

#MODULES=essential arguments objects maths logging extensions io cume
#MODULES=essential arguments objects maths logging extensions
MODULES=essential objects maths

# -------------------------------------------------------------------
# Main variables
# -------------------------------------------------------------------

MAKE=make --no-print-directory
TODAY=$(shell date '+%Y_%m_%d')
ARCHIVE="$(HOME)/export/$(MY_PROJECT_NAME)_$(TODAY).tgz"

# -------------------------------------------------------------------
# Directories
# -------------------------------------------------------------------

# Directory of the project
PROJECT_DIR=$(shell pwd)

# where all compiled files are send
BUILD_DIR=$(PROJECT_DIR)/build/$(MY_PROJECT_VERSION)

# Source files of the library
LIB_SRC_DIR=$(PROJECT_DIR)/src/$(MY_PROJECT_VERSION)/lib/
# Directory of related examples
XMP_SRC_DIR=$(PROJECT_DIR)/src/$(MY_PROJECT_VERSION)/xmp/
# Directory of regression tests
TST_SRC_DIR=$(PROJECT_DIR)/src/$(MY_PROJECT_VERSION)/tst/

# Directory of C/C++ include files (.h)
INCLUDES_DIR=$(BUILD_DIR)/include

# Flavor / Architecture / Compiler
FLARCO=$(MY_FLAVOR)_$(MY_ARCHITECTURE)_$(MY_COMPILER)


INSTALL_DIR=/usr/local
INSTALL_INCLUDE_DIR=$(INSTALL_DIR)/include/ez/$(MY_PROJECT_VERSION)
INSTALL_LIBRARY_DIR=$(INSTALL_DIR)/lib/ez/$(MY_PROJECT_VERSION)/$(FLARCO)

RUN_XMP_DIR=$(BUILD_DIR)/$(FLARCO)/xmp
RUN_XMP_OUTPUT=$(RUN_XMP_DIR)/run_examples.txt


# -------------------------------------------------------------------
# Rules
# -------------------------------------------------------------------

.SUFFIXES: .o .cpp

all: introduction information create_directories compile

introduction:
	cat app/ezlib_logo.txt

information:
	@echo " "
	@echo "//####################################################################"
	@echo "// PROJECT  $(MY_PROJECT_NAME) $(MY_PROJECT_VERSION)"
	@echo "//  AUTHOR  $(MY_AUTHOR_NAME)"
	@echo "//   EMAIL  $(MY_AUTHOR_EMAIL)"
	@echo "// ___________________________________________________________________"
	@echo "// System       = $(MY_SYSTEM)"
	@echo "// Flavor       = $(MY_FLAVOR)"
	@echo "// Compiler     = $(MY_COMPILER)"
	@echo "//####################################################################"
	@echo " "

create_directories:
	@mkdir -p $(BUILD_DIR) $(INCLUDES_DIR)

# -------------------------------------------------------------------
# Use 'make config' or 'make configure' before to compile modules
# -------------------------------------------------------------------
config:
	app/scripts/configure.sh

configure:
	app/scripts/configure.sh


# -------------------------------------------------------------------
# Compilation of modules into objects (.o) and libraries (.so, .a)
# use 'make config' before to compile
# -------------------------------------------------------------------
compile:
	@for module in $(MODULES) ; do \
		echo "- copy include files of $$module to $(INCLUDES_DIR)" ; \
		cd $(LIB_SRC_DIR)/$$module ; \
		$(MAKE) copy_include_files PROJECT_DIR=$(PROJECT_DIR) BUILD_DIR=$(BUILD_DIR) MODULE=$$module ; \
		cd $(PROJECT_DIR) ; \
	done
	@for module in $(MODULES) ; do \
		echo "************************************************************" ; \
		echo "*" ; \
		echo "* BUILD MODULE $$module" ; \
		echo "*" ; \
		echo "************************************************************" ; \
		cd $(LIB_SRC_DIR)/$$module ; \
		$(MAKE) compile PROJECT_DIR=$(PROJECT_DIR) BUILD_DIR=$(BUILD_DIR) MODULE=$$module ; \
		cd $(PROJECT_DIR) ; \
	done


# -------------------------------------------------------------------
# Compile examples
# -------------------------------------------------------------------
examples:
	@for module in $(MODULES) ; do \
		echo "************************************************************" ; \
		echo "*" ; \
		echo "GENERATE EXAMPLES FOR $$module" ; \
		echo "*" ; \
		echo "************************************************************" ; \
		cd $(XMP_SRC_DIR)/$$module ; \
		$(MAKE) PROJECT_DIR=$(PROJECT_DIR) BUILD_DIR=$(BUILD_DIR) MODULE=$$module ; \
		if test $$? -ne 0 ; then \
			cd $(PROJECT_DIR) ; \
			exit 1 ; \
		fi ; \
		cd $(PROJECT_DIR) ; \
	done

# -------------------------------------------------------------------
# compile cuda
# -------------------------------------------------------------------
cuda:
	@for module in $(CUDA_MODULES) ; do \
		echo "- copy include for MODULE $$module" ; \
		cd $(SRC_DIR)/version_$(MY_PROJECT_VERSION)/$$module ; \
		$(MAKE) copy_includes PROJECT_DIR=$(PROJECT_DIR) BUILD_DIR=$(BUILD_DIR) MODULE=$$module ;  \
		cd $(PROJECT_DIR) ; \
	done
	@for module in $(CUDA_MODULES) ; do \
		echo "===============" ; \
		echo "MODULE $$module" ; \
		echo "===============" ; \
		cd $(SRC_DIR)/version_$(MY_PROJECT_VERSION)/$$module ; \
		$(MAKE) PROJECT_DIR=$(PROJECT_DIR) BUILD_DIR=$(BUILD_DIR) MODULE=$$module ; \
		if test $$? -ne 0 ; then \
			cd $(PROJECT_DIR) ; \
			exit 1 ; \
		fi ; \
		cd $(PROJECT_DIR) ; \
	done

.PHONY: tests

# -------------------------------------------------------------------
# compile tests
# -------------------------------------------------------------------
tests:
	@for module in $(MODULES) ; do \
		echo "============================================================" ; \
		echo "COMPILE TESTS $$module" ; \
		echo "============================================================" ; \
		cd $(TST_SRC_DIR)$$module ; \
		$(MAKE) PROJECT_DIR=$(PROJECT_DIR) BUILD_DIR=$(BUILD_DIR) MODULE=$$module ; \
		if test $$? -ne 0 ; then \
			cd $(PROJECT_DIR) ; \
			exit 1 ; \
		fi ; \
		cd $(PROJECT_DIR) ; \
	done

# -------------------------------------------------------------------
# compile cuda tests
# -------------------------------------------------------------------
cuda_tests:
	@for module in $(CUDA_MODULES) ; do \
		echo "===============" ; \
		echo "TESTS $$module" ; \
		echo "===============" ; \
		cd $(TESTS_DIR)/version_$(MY_PROJECT_VERSION)/$$module ; \
		$(MAKE) PROJECT_DIR=$(PROJECT_DIR) BUILD_DIR=$(BUILD_DIR) MODULE=$$module ; \
		if test $$? -ne 0 ; then \
			cd $(PROJECT_DIR) ; \
			exit 1 ; \
		fi ; \
		cd $(PROJECT_DIR) ; \
	done

# -------------------------------------------------------------------
# Execute tests
# -------------------------------------------------------------------
run_tests:
	app/scripts/run_tests.sh $(PROJECT_DIR) $(BUILD_DIR)/tests/$(FLARCO) $(RUN_TESTS_DIR) $(RUN_TESTS_FILE) $(MODULES)

.PHONY: examples


# -------------------------------------------------------------------
# Run examples
# -------------------------------------------------------------------
run_examples:
	@date > $(RUN_EXAMPLES_OUTPUT)
	@for module in $(MODULES) ; do \
		echo "============================================================" ; \
		echo "************************************************************" ; \
		echo "*" ; \
		echo "* RUN EXAMPLES FOR $$module" ; \
		echo "*" ; \
		echo "************************************************************" ; \
		echo "************************************************************" >>$(RUN_EXAMPLES_OUTPUT) ; \
		echo "*" >>$(RUN_EXAMPLES_OUTPUT) ; \
		echo "* RUN EXAMPLES FOR $$module" >>$(RUN_EXAMPLES_OUTPUT) ; \
		echo "*" >>$(RUN_EXAMPLES_OUTPUT) ; \
		echo "************************************************************" >>$(RUN_EXAMPLES_OUTPUT) ; \
		cd $(XMP_SRC_DIR)/$$module ; \
		$(MAKE) run_examples PROJECT_DIR=$(PROJECT_DIR) BUILD_DIR=$(BUILD_DIR) MODULE=$$module OUTPUT=$(RUN_EXAMPLES_OUTPUT); \
		if test $$? -ne 0 ; then \
			cd $(PROJECT_DIR) ; \
			exit 1 ; \
		fi ; \
		cd $(PROJECT_DIR) ; \
	done
	@echo "============================================================"
	@echo "Output in file $(RUN_EXAMPLES_OUTPUT)"


# -------------------------------------------------------------------
# Compile cuda examples
# -------------------------------------------------------------------
cuda_examples:
	@cd $(CUDA_EXAMPLES_DIR)/version_$(MY_PROJECT_VERSION) ; \
	$(MAKE) PROJECT_DIR=$(PROJECT_DIR) BUILD_DIR=$(BUILD_DIR) ;\
	cd $(PROJECT_DIR)


# -------------------------------------------------------------------
# Clean modules
# -------------------------------------------------------------------
clean: information
	@for module in $(MODULES) ; do \
		echo "- clean $$module" ; \
		cd $(LIB_SRC_DIR)/$$module ; \
		$(MAKE) clean PROJECT_DIR=$(PROJECT_DIR) BUILD_DIR=$(BUILD_DIR) MODULE=$$module ; \
		cd $(PROJECT_DIR) ; \
	done

# -------------------------------------------------------------------
# Clean modules
# -------------------------------------------------------------------
deep_clean: information
	@echo "- deep clean, remove all files in $(BUILD_DIR)" ;\
	rm -rf $(BUILD_DIR)


# -------------------------------------------------------------------
# Clean tests
# -------------------------------------------------------------------
clean_tests:
	@echo "- clean tests directory by removing all files" ;\
	rm -rf $(TESTS_DIR)

# -------------------------------------------------------------------
# Clean examples
# -------------------------------------------------------------------
clean_examples:
	@echo "- clean examples directory by removing all files in"
	@for module in $(MODULES) ; do \
		echo "============================================================" ; \
		echo "CLEAN EXAMPLES FOR $$module" ; \
		echo "============================================================" ; \
		cd $(XMP_SRC_DIR)/$$module ; \
		$(MAKE) clean PROJECT_DIR=$(PROJECT_DIR) BUILD_DIR=$(BUILD_DIR) MODULE=$$module ; \
		if test $$? -ne 0 ; then \
			cd $(PROJECT_DIR) ; \
			exit 1 ; \
		fi ; \
		cd $(PROJECT_DIR) ; \
	done



# -------------------------------------------------------------------
# Install include files and libraries
# -------------------------------------------------------------------
install:
	@mkdir -p $(INSTALL_INCLUDE_DIR)
	@mkdir -p $(INSTALL_LIBRARY_DIR)
	@echo "- copy include files to $(INSTALL_INCLUDE_DIR)"
	@cp -R $(BUILD_DIR)/include/* $(INSTALL_INCLUDE_DIR)
	@echo "- copy libraries to $(INSTALL_LIBRARY_DIR)/lib"
	@cp $(BUILD_DIR)/$(FLARCO)/lib/lib_essential.a  $(INSTALL_LIBRARY_DIR)/libez_essential.a
	@cp $(BUILD_DIR)/$(FLARCO)/lib/lib_extensions.a $(INSTALL_LIBRARY_DIR)/libez_extensions.a
	@cp $(BUILD_DIR)/$(FLARCO)/lib/lib_logging.a 	 $(INSTALL_LIBRARY_DIR)/libez_logging.a
	@cp $(BUILD_DIR)/$(FLARCO)/lib/lib_arguments.a  $(INSTALL_LIBRARY_DIR)/libez_arguments.a
	@cp $(BUILD_DIR)/$(FLARCO)/lib/lib_maths.a 	 $(INSTALL_LIBRARY_DIR)/libez_maths.a
	@cp $(BUILD_DIR)/$(FLARCO)/lib/lib_objects.a 	 $(INSTALL_LIBRARY_DIR)/libez_objects.a
	@cp $(BUILD_DIR)/$(FLARCO)/lib/lib_io.a 		 $(INSTALL_LIBRARY_DIR)/libez_io.a
	@cp $(BUILD_DIR)/$(FLARCO)/lib/lib_cume.a 		 $(INSTALL_LIBRARY_DIR)/libez_cume.a
	@ls $(INSTALL_LIBRARY_DIR)/libez*

# -------------------------------------------------------------------
# generate documentation
# use doxygen -g to generate config file "Doxyfile"
# -------------------------------------------------------------------
doc:
	@echo "To generate the documentation you need to have"
	@echo "- 'doxygen' and 'dot' installed"
	@echo "Install with:"
	@echo "sudo apt install doxygen graphviz"
	@echo " "
	doxygen docxyfile

# -------------------------------------------------------------------
# generate archive from all files in the project folder
# -------------------------------------------------------------------
archive: deep_clean
	@echo "- generate archive " ;\
	mkdir -p ~/export ;\
	date +'%Y/%m/%d %Hh%M' >timestamp ;\
	echo "- archive file:\n$(ARCHIVE)" ;\
	mkdir -p `dirname $(ARCHIVE)` ;\
	cd .. ; \
	tar -czf $(ARCHIVE) $(MY_PROJECT_NAME) ; \
	cd $(PROJECT_DIR)
