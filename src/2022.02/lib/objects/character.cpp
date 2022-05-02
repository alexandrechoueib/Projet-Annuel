/*
 * character.cpp
 *
 *  Created on: Jul 28, 2017
 *      Author: richer
 */

#include "objects/character.h"

using namespace ez::objects;

std::ostream& Character::print(std::ostream& stream) {
	stream << m_value;
	return stream;
}


character Character::_char_(boolean b) {
	return (b == false) ? (character) '\0' : (character) '\1';
}

character Character::_char_(integer i) {
	return static_cast<character>(i % 256);
}

character Character::_char_(real r) {
	return static_cast<character>(static_cast<int>(r) % 256);
}

character Character::_char_(text t) {
	if (t.length() == 0) return '\0';
	return t[0];
}
