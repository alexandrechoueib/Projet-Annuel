/*
 * integer_argument.cpp
 *
 *  Created on: May 7, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "arguments/integer_argument.h"

using namespace ez::arguments;


void IntegerArgument::parse(text& s) {

	if (s.length() == 0) {
		notify( "integer value not provided" );
	} else {
		if (eze::Types::is_integer( s )) {
			*_value = eze::Types::to_integer( s );
			
		} else {
			notify( "value '" << s << "' is not an integer value" );
		}
	}

}
