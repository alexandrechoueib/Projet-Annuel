/*
 * natural_range_argument.h
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

#ifndef ARGUMENTS_NATURAL_RANGE_ARGUMENT_H_
#define ARGUMENTS_NATURAL_RANGE_ARGUMENT_H_

#include "arguments/natural_argument.h"
using namespace ez::essential;

// name space for Command Line Argument Parser
namespace ez {

namespace arguments {

/**
 * class used to treat natural range argument
 */
class NaturalRangeArgument : public NaturalArgument {
protected:
	natural _min_value, _max_value;

public:
	NaturalRangeArgument(string long_label, char short_label,
			natural *value, natural mini, natural maxi,
			string description, trigger_t trigger = nullptr)
		: NaturalArgument( long_label, short_label, value, description, trigger ) {
		
		_min_value = mini;
		_max_value = maxi;
		
	}


	void parse( text& s );


	natural min() { 
	
		return _min_value; 
		
	}
	
	
	natural max() { 
	
		return _max_value; 
	
	}
};


} // end of namespace arguments

} // end of namespace ez

#endif /* ARGUMENTS_NATURAL_RANGE_ARGUMENT_H_ */
