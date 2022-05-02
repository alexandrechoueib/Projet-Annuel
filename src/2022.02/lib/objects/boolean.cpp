/*
 * boolean.cpp
 *
 *  Created on: Apr 11, 2017
 *      Author: Jean-Michel Richer
 */


#include "objects/boolean.h"

using namespace ez::objects;

boolean Boolean::zero = false;
Boolean Boolean::zero_object(false);

/**
 * overloading of print method
 */
std::ostream& Boolean::print(std::ostream& stream) {
	if (m_value)
		stream << "true";
	else
		stream << "false"; 
	return stream;	
}


bool Boolean::_bool_(character c) {
	return (c == '\0') ? false : true;
}

bool Boolean::_bool_(integer i) {
	return (i == 0) ? false : true;
}

bool Boolean::_bool_(real r) {
	return (r == 0.0) ? false : true;
}

bool Boolean::_bool_(text s) {
	if (s == "true") return true;
	else if (s == "false") return false;
	// any of true/false was found so raise exception
	notify("true or false expected");
	return false;
}
