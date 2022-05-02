/*
 * argument.cpp
 *
 *  Created on: May 7, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "arguments/argument.h"

using namespace ez::arguments;

void Argument::call_trigger() {

	if (_trigger != nullptr) {
		(*_trigger)();
	}
	
}


