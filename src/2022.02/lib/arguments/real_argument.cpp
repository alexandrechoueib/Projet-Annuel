/*
 * real_argument.cpp
 *
 *  Created on: May 7, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "arguments/real_argument.h"

using namespace ez::arguments;


void RealArgument::parse( text& s ) {

	if (s.length() == 0) {
		notify( "floating point value not provided" );
		
	} else {
		if (ez::essential::Types::is_real( s )) {
			*_value = atof( s.c_str() );
			
		} else {
			notify( "value '" << s << "' is not a floating point value" );
		}
	}
}


