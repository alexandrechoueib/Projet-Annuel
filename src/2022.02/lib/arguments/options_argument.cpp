/*
 * options_argument.cpp
 *
 *  Created on: May 7, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "arguments/options_argument.h"

using namespace ez::arguments;


OptionsArgument::OptionsArgument( string long_label, char short_label,
		natural *value, vector<string>& options,
		string description, trigger_t trigger )
		: Argument( long_label, short_label, description, trigger )  {
		
	_value = value;
	_options = options;
	
	if (options.size() <= 2) {
		notify( "Option definition needs more than one argument" );
	}

}


void OptionsArgument::parse( text& s ) {

	if (s.length() == 0) {
		notify( "optional value not provided" );
		
	} else {
		*_value = find_option(s);
		if (*_value > _options.size()) {
			string short_option = "";
			
			if (_short_label != '\0') {
				short_option = " or -";
				short_option += _short_label;
				short_option += " " + *_value;
			}
	
			notify( " parse option error ! " << "option label --" 
				<< _long_label << "=" << *_value << " must be chosen in ["
				<< get_allowed_options() << "]" );
		}
	}
	
}


natural OptionsArgument::find_option( text& s ) {

	for (natural i = 0; i < _options.size(); ++i) {
		if (_options[i] == s) return i+1;
	}
	
	return _options.size() + 1;
	
}


text OptionsArgument::get_allowed_options() {

	text s = _options[ 0 ];

	for (natural i = 1; i < _options.size(); ++i) {
		s += ", ";
		s += _options[ i ];
	}

	return s;
	
}


