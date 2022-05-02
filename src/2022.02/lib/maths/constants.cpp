/*
 * constants.cpp
 *
 *  Created on: Apr 11, 2017
 *      Author: Jean-Michel Richer
 */

#include "maths/constants.h"

namespace eze = ez::essential;
using namespace ez::maths;

eze::real Constants::REAL_EPSILON = 1e-11;

const eze::real Constants::ANGLE_DEGREE_0 = 0;
const eze::real Constants::ANGLE_DEGREE_45 = M_PI / 4.0;
const eze::real Constants::ANGLE_DEGREE_90 = M_PI / 2.0;
const eze::real Constants::ANGLE_DEGREE_180 = M_PI;
const eze::real Constants::ANGLE_DEGREE_270 = 3.0 * M_PI / 2.0;
const eze::real Constants::REAL_MIN = -1e+308;
const eze::real Constants::REAL_MAX = 1e+308;
