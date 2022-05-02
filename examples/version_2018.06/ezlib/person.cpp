/*
 * person.cpp
 *
 *  Created on: Jul 31, 2017
 *      Author: richer
 */

#include "../../examples/version_2018.06/person.h"

Person::Person() : Object(), m_name(""), m_age(-1) {

}

Person::Person(text name, int age) : Object(), m_name(name), m_age(age) {

}

Person::Person(const self& obj) {
	m_name = obj.m_name;
	m_age = obj.m_age;
}

Person& Person::operator=(const self& obj) {
	if (&obj != this) {
		m_name = obj.m_name;
		m_age = obj.m_age;
	}
	return *this;
}

Person::~Person() {

}

