/*
 * text_argument.cpp
 *
 *  Created on: May 7, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "arguments/text_argument.h"

using namespace ez::arguments;

void TextArgument::parse( text& s ) {

	*_value = s;

}
