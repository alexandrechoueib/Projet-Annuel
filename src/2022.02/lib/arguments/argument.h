/*
 * argument.h
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

#ifndef ARGUMENTS_ARGUMENT_H_
#define ARGUMENTS_ARGUMENT_H_

#include <map>
#include <vector>
#include <typeinfo>
#include <algorithm>

#include "essential/import.h"

using namespace std;
using namespace ez::essential;

// name space for Command Line Argument Parser
namespace ez {

namespace arguments {

class ArgumentsParser;
class Argument;

typedef void (*trigger_t)(void);

/**
 * default pattern to declare one class of argument
 */
class Argument {
protected:

	// ==========================================
	// Data members
	// ==========================================

	// description as long label
	text _long_label;
	// description as short label
	char _short_label;
	// full description of argument
	text _description;
	// true if this argument is needed
	bool _required;
	// true if option was found when parsing command
	// line arguments
	bool _found;
	// if true will ne be reported in the synopsis
	bool _hidden;
	// function that needs to be called when parsing argument
	trigger_t _trigger;


public:
	/**
	 * constructor with labels and description
	 * @param ll label as long format (text)
	 * @param sl label as short format (char)
	 * @param d description
	 * @param tg trigger
	 */
	Argument( text long_label, char short_label, 
		text description,
		trigger_t trigger = nullptr) {
		
		_long_label = long_label;
		_short_label = short_label;
		_description = description;
		_required = false;
		_found = false;
		_trigger = trigger;
		_hidden = false;
		
	}

	/**
	 * constructor with labels and description but without short argument
	 * @param ll label as long format (text)
	 * @param d description
	 * @param tg trigger
	 */
	Argument( text long_label, text description,
			trigger_t trigger = nullptr) {
			
		_long_label = long_label;
		_short_label = '\0';
		_description = description;
		_required = false;
		_found = false;
		_trigger = trigger;
		_hidden = false;
		
	}


	virtual ~Argument() {

	}

	// ==========================================
	// Getters
	// ==========================================
	
	text get_long_label() {

		return _long_label;

	}


	char get_short_label() {

		return _short_label;

	}


	text get_description() {

		return _description;

	}


	void hidden( bool value ) {
	
		_hidden = value;
		
	}


	bool hidden() {
	
		return _hidden;
		
	}

	/**
	 * function used to parse argument argv[i].
	 * for example if argument is "--arg=value", only the "value"
	 * part is given as the s parameter
	 */
	virtual void parse( text& s ) = 0;

	/**
	 * tells if this argument is required
	 */
	bool is_required() {
	
		return _required;
		
	}


	void set_required() {
	
		_required = true;
		
	}


	void set_found() {
	
		_found = true;
		
	}


	bool was_found() {
		
		return _found;
		
	}

	/**
	 * call trigger if defined
	 */
	void call_trigger();
	

	friend class ArgumentParser;
};


} // end of namespace arguments

} // end of namespace ez



#endif /* ARGUMENTS_ARGUMENT_H_ */
