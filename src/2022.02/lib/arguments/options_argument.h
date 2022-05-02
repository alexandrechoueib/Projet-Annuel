/*
 * options_argument.h
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

#ifndef ARGUMENTS_OPTIONS_ARGUMENT_H_
#define ARGUMENTS_OPTIONS_ARGUMENT_H_

#include "arguments/argument.h"
using namespace ez::essential;

// name space for Command Line Argument Parser
namespace ez {

namespace arguments {

/**
 * class used to treat user defined or label argument.
 * A label argument will result as an integer value to set
 * in function of a list of labels
 */
class OptionsArgument : public Argument {
protected:
	natural *_value;
	// array of allowed labels
	vector<string> _options;

public:
	OptionsArgument( string long_label, char short_label,
			natural *value, vector<string>& options,
			string description, trigger_t trigger = nullptr );


	void parse( text& s );


	natural find_option( text& s );


	text get_allowed_options();

};


} // end of namespace arguments

} // end of namespace ez


#endif /* ARGUMENTS_OPTIONS_ARGUMENT_H_ */
