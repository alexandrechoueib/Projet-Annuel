/*
 * argument_parser.h
 *
 *  Created on: May 7, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

/*
    EZLib version 2022.02
    Copyright (C) 2019-2022  Jean-Michel Richer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

*/

#ifndef ARGUMENTS_ARGUMENT_PARSER_H_
#define ARGUMENTS_ARGUMENT_PARSER_H_

#include "arguments/boolean_argument.h"
#include "arguments/flag_argument.h"
#include "arguments/integer_argument.h"
#include "arguments/integer_range_argument.h"
#include "arguments/natural_argument.h"
#include "arguments/natural_range_argument.h"
#include "arguments/options_argument.h"
#include "arguments/real_argument.h"
#include "arguments/real_range_argument.h"
#include "arguments/text_argument.h"
#include "essential/import.h"

using namespace ez::essential;

// name space for Command Line Argument Parser
namespace ez {

namespace arguments {

/**
 * main class to treat command line arguments of argv
 */
class ArgumentParser {
protected:

	std::vector<string> _command;

	// list of arguments
	vector<Argument *> _arguments;

	// direct access to arguments using short or long format
	map<text, Argument *> _map_long;
	map<char, Argument *> _map_short;

	// name of program
	text _program_name;
	// short description of what program does
	text _program_desc;

	// default help argument
	Argument *_help_command;
	// default output argument
	Argument *_output_command;

	// show synopsys of all arguments
	bool _show_synopsis;

public:
	/**
	 * constructor with argc and argv
	 * @param program_name name of program
	 * @param description description of what the program does
	 * @param argc number of command line arguments
	 * @param argv array of text of command line arguments
	 */
	ArgumentParser( text program_name, text description, 
		int argc, char *argv[] );

	/**
	 * constructor with vector of string as command line arguments
	 */
	ArgumentParser( text program_name, text description, 
		std::vector<string>& command );

	/**
	 * destructor
	 */
	~ArgumentParser();

	/**
	 * remove all created arguments
	 */
	void clean();

	/**
	 * add new type of argument
	 * we check if the new argument to add does not have a short or long
	 * label equal to one of the existing arguments
	 * if it is a LabelArgument we check if all options are different
	 * @param arg pointer to new argument
	 */
	Argument *add( Argument *arg );

	Argument *add_flag   ( text long_label, char short_label, bool *value, text description, trigger_t trigger = nullptr );
	Argument *add_boolean( text long_label, char short_label, bool *value, text description, trigger_t trigger = nullptr );
	Argument *add_integer( text long_label, char short_label,  integer *value, text description, trigger_t trigger = nullptr );
	Argument *add_natural( text long_label, char short_label,  natural *value, text description, trigger_t trigger = nullptr );
	Argument *add_real   ( text long_label, char short_label,  real *value, text description, trigger_t trigger = nullptr );
	Argument *add_real_range( text long_label, char short_label,  real *value, real mini, real maxi, text description, trigger_t trigger = nullptr );
	Argument *add_integer_range( text long_label, char short_label,  integer *value, integer mini, integer maxi, text description, trigger_t trigger = nullptr );
	Argument *add_natural_range( text long_label, char short_label,  natural *value, natural mini, natural maxi, text description, trigger_t trigger = nullptr );
	Argument *add_text ( text long_label, char short_label, text *value, text description, trigger_t trigger = nullptr );
	Argument *add_options( text long_label, char short_label, natural *value, vector<text>& options, text description, trigger_t trigger = nullptr );


	natural nbr_arguments() {
	
		return _arguments.size();
		
	}

	/**
	 * parse argv
	 * @param remaining will contain the arguments that where not
	 * 		recognized as options starting by '-' or '--'
	 * @param max_remaining tells the maximum number of arguments
	 * 		that are allowed when they are not recognized as an
	 * 		option that starts by '-' or '--'. If equals to zero
	 * 		then no remaining argument is allowed and an exception
	 * 		will be thrown
	 * throw Exception in case of error
	 */
	void parse( std::vector<string>& remaining, natural max_remaining = 0 );

	/**
	 * report error
	 */
	void report_error( exception& e, bool show_synopsis = true );

protected:

	/**
	 * print synopsis of command
	 * @param out output stream
	 * @param prog_name name of program
	 * @param descr description of program (ex for ls : list directory contents)
	 */
	void print_synopsis( std::ostream& out );


	/**
	 * print text with given display length
	 */
	void print_synopsis_text( std::ostream& out, integer level, text s, natural length );

	/**
	 * return if we can print synopsis
	 */
	bool has_to_show_synopsis() {
	
		return _show_synopsis;
		
	}

	/**
	 * tell if we can show synopsis
	 */
	void set_has_to_show_synopsis( bool v ) {
	
		_show_synopsis = v;
		
	}

};


} // end of namespace arguments

} // end of namespace ez




#endif /* VERSION_2017_04_ARGUMENTS_ARGUMENT_PARSER_H_ */
