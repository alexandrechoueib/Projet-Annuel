/*
 * real_range_argument.cpp
 *
 *  Created on: May 7, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "arguments/real_range_argument.h"

using namespace ez::arguments;

void RealRangeArgument::parse( text& s ) {

	if (s.length() == 0) {
		notify( "floating point value not provided" );
		
	} else {
		if (ez::essential::Types::is_real( s )) {
			*_value = atof( s.c_str() );
			if ((*_value < _min_value) || (*_value > _max_value)) {
				notify("value given as parameter --" << _long_label << "=" << *_value
					<< " or -" << _short_label << " " << *_value
					<< " is not in float range [" << _min_value << ".." << _max_value << "]");
			}
			
		} else {
			notify( "value '" << s << "' is not a floating point value" );
		}
	}
}




