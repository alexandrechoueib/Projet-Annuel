/*
 * numeric.h
 *
 *  Created on: May 1, 2017
 *      Author: Jean-Michel Richer
 */

#ifndef OBJECTS_NUMERIC_H_
#define OBJECTS_NUMERIC_H_

#include <numeric>
#include "essential/types.h"
#include "objects/integer.h"
#include "objects/long_integer.h"
#include "objects/long_natural.h"
#include "objects/natural.h"
#include "objects/real.h"

namespace ez {

namespace objects {

template<class T>
bool is_numeric() {
	if (scalar_is_numeric<T>()) return true;
	if (typeid(T) == typeid(Integer)) return true;
	if (typeid(T) == typeid(LongInteger)) return true;
	if (typeid(T) == typeid(Natural)) return true;
	if (typeid(T) == typeid(LongNatural)) return true;
	if (typeid(T) == typeid(Real)) return true;
	return false;
}

} // end of namespace objects

} // end of namespace ez

#endif /* OBJECTS_NUMERIC_H_ */
