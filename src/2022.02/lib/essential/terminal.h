/*
 * terminal.h
 *
 *  Created on: May 25, 2015
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

/*
    EZLib version 2022.02
    Copyright (C) 2019-2022  Jean-Michel Richer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

*/

#ifndef ESSENTIAL_TERMINAL_H_
#define ESSENTIAL_TERMINAL_H_

#include <string>
#include <iostream>
#include "essential/scalar_types.h"

using namespace std;

namespace ez {

namespace essential {

class Terminal {
public:
	// line types
	static string line_type_1;
	static string line_type_2;
	static string line_type_3;
	static string line_type_4;
	static string line_type_5;

	// begin underline
	static string b_underline;
	// end underline
	static string e_underline;
	// begin bold
	static string b_bold;
	// end bold
	static string e_bold;

	/**
	 * Convert string to bold for terminal display
	 */
	static text bold( text s );
	
	/**
	 * Convert string to underline for terminal display
	 */
	static text underline( text s );

	/**
	 * Wait for user to press Return key on terminal
	 */
	static void press_return();
	
};

} // end of namespace essential

} // end of namespace ez

#endif /* ESSENTIAL_TERMINAL_H_ */
