/*
 * natural_argument.cpp
 *
 *  Created on: May 7, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "arguments/natural_argument.h"

using namespace ez::arguments;

void NaturalArgument::parse(text& s) {

	if (s.length() == 0) {
		notify( "natural value not provided" );
		
	} else {
		if (eze::Types::is_natural( s )) {
			*_value = eze::Types::to_natural( s );
		} else {
			notify( "value '" << s << "' is not a natural value" );
		}
	}
	
}



