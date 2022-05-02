/*
 * how_to_arguments_xmp.cpp
 *
 *  Created on: Aug 19, 2021
 *      Author: richer
 */

#include "arguments/import.h"
#include "essential/import.h"

natural _natural = 0;

integer _integer = 0;

boolean _boolean = false;

real _real = 0.0;

integer _integer_range = 10;
integer _integer_range_min = 10;
integer _integer_range_max = 20;

std::vector<text> options = { "one", "two", "three", "four" };
natural _option;

/**
 * Program entry. 
 */  
int main(int argc, char *argv[]) {
	eza::ArgumentParser parser(argv[0], "how to use the class ArgumentParser", argc, argv);

	std::vector<text> remaining_arguments; 
	 
	try {
		parser.add_natural("natural", 'n', &_natural, "natural");
		parser.add_integer("integer", 'i', &_integer, "integer");
		parser.add_boolean("boolean", 'b', &_boolean, "boolean");
		parser.add_real("real", 'r', &_real, "real");
		parser.add_integer_range("integer_range", 'x', &_integer_range, 
			_integer_range_min,
			_integer_range_max,
			"integer range");
		parser.add_options("option", 'y', &_option, options, "options: one, two, three, four");
		
		parser.parse(remaining_arguments);
		
	} catch(Exception& e) {
		parser.report_error(e);
		return EXIT_FAILURE;
	}
 
	cout << "boolean=" << _boolean << endl;
	cout << "integer=" << _integer << endl;
	cout << "natural=" << _natural << endl;
	cout << "real   =" << _real << endl;
	cout << "range  =" << _integer_range << endl;
	cout << "option =" << _option << endl;    
	return EXIT_SUCCESS;
}


