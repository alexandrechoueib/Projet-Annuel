/*
 * boolean_argument.cpp
 *
 *  Created on: May 7, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "arguments/boolean_argument.h"

using namespace ez::arguments;


void BooleanArgument::parse( text& s ) {

	if (s.length() == 0) {
		notify( "boolean value not provided" );
	} else {
		if (s == "true") {
			*_value = true;
		} else if (s == "false") {
			*_value = false;
		} else {
			notify( "value '" << s << "' is not a boolean value, use 'true' or 'false'" );
		}
	}
	
}

