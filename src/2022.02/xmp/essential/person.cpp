/*
 * person.cpp
 *
 *  Created on: Jul 31, 2017
 *      Author: richer
 */

#include "person.h"

Person::Person() : Object(), _name(""), _age(-1) {

}

Person::Person( text name, int age ) : Object(), _name( name ), _age( age ) {

}

Person::Person( const self& obj ) {

	_name = obj._name;
	_age = obj._age;

}


Person& Person::operator=( const self& obj ) {

	if (&obj != this) {
		_name = obj._name;
		_age = obj._age;
	}
	return *this;

}


Person::~Person() {

}

