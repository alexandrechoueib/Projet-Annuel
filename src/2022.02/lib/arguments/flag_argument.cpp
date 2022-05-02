/*
 * flag_argument.cpp
 *
 *  Created on: May 7, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "arguments/flag_argument.h"

using namespace ez::arguments;


void FlagArgument::parse( text& s) {

	if (s.length() != 0) {
		notify( "argument not allowed" );
	}
	
	*_value = true;
	
}

