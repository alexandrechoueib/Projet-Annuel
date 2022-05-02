/*
 * object.cpp
 *
 *  Created on: Apr 15, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "objects/object.h"

using namespace ez::objects;

text cause_error_message = " is not implemented in "
		"the class that you are using.";
text remedy_error_message = "You need to overload it "
		"in the class from where it was called";


std::ostream& Object::print(std::ostream& stream) {
	notify("the print method" << cause_error_message << remedy_error_message);
	return stream;
}


integer Object::compare(const Object& y) {
	notify("the compare method" << cause_error_message << remedy_error_message);
}

Object *Object::clone() {
	notify("the clone method" << cause_error_message << remedy_error_message);
	return nullptr;
}

