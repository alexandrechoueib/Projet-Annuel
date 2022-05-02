/*
 * text_utils.h
 *
 *  Created on: Apr 2, 2015
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

#ifndef ESSENTIAL_TEXT_UTILS_H_
#define ESSENTIAL_TEXT_UTILS_H_

#include <functional>
#include <cctype>
#include <algorithm>
#include <vector>
#include <cstring>
#include <limits.h>
#include <iomanip>

#include "essential/ensure.h"
#include "essential/types.h"

using namespace std;

namespace ez {

namespace essential {

#define TEXT_BLANKS " \f\n\r\t\v"
#define TEXT_BLANK  " "

class TextUtils {
public:

	/**
	 * print each character of text as ascii code and character if its
	 * ascii code is greater than 32 (' ')
	 */
	static void inspect( std::ostream& out, text& s );

	/**
	 * remove all characters that correspond to any character of pattern
	 * at the end of the text
	 */
	static void trim_right( text& s, const text& pattern = TEXT_BLANKS );

	/**
	 * remove all characters that correspond to any character of pattern
	 * at beginning of the text
	 */
	static void trim_left( text& s, const text& pattern = TEXT_BLANKS );

	/**
	 * remove all characters that correspond to any character of pattern
	 * at beginning and end of the text
	 */
	static void trim( text& s, const text& pattern = TEXT_BLANKS );

	/**
	 * erase all characters that correspond to any character of pattern
	 */
	static void erase( text& s, const text& pattern = TEXT_BLANKS );

	/**
	 * remove all spaces characters
	 */
	static void remove_spaces( text& s );

	/**
	 * replace all occurrences of source character by replacement character
	 */
	static void replace_char( text& s, char src, char repl );

	/**
	 * replace 'pattern' by 'replace' string
	 */
	static void replace( text& s, string pattern, string replace );

	/**
	 * convert text to lowercase
	 */
	static void lower( text& s );

	/**
	 * convert text to uppercase
	 */
	static void upper( text& s );

	/**
	 * convert text represented in C format to lowercase
	 */
	static void lower( char *s );

	/**
	 * convert text represented in C format to uppercase
	 */
	static void upper( char *s );

	/**
	 * split text in subtexts when separated by any character of delimiters
	 * the subtexts are put in a vector
	 * @param s text to explode
	 * @param v output subtexts
	 * @param delim characters used as delimiters
	 */
	static void explode( text& s, std::vector<text>& v, 
		const text& delim = TEXT_BLANKS );

	/**
	 * join texts of a vector into a single text where delimiters separate
	 * each subtext
	 * @param s text used as a result
	 * @param v vector of texts to implode
	 *
	 */
	static void implode( text& s, std::vector<text>& v, 
		const text& delim = TEXT_BLANK);
		
	static void implode( text& s, std::vector<char>& v, 
		const text& delim = TEXT_BLANK);

	/**
	 * return true if st_str is found in s
	 */
	static bool contains( text& s, text& st_str );

	/**
	 * return position of st_str in s or text::npos if not found
	 * @param s text that will be searched
	 * @param st_str subtext searched in s
	 * @return position of st_str in s where first position is 1. If the st_str text
	 * is not found we return 0.
	 */
	static size_t position_of( text& s, text& st_str );

	/**
	 * return position of st_str in s or text::npos if not found
	 * @param s text that will be searched
	 * @param st_str subtext searched in s
	 * @return position of st_str in s where first position is 1. If the st_str text
	 * is not found we return 0.
	 */
	static size_t position_of( text& s, char *st_str );

	/**
	 * remove length characters from given pos
	 * @param pos position where to start removing (where first character is at position 1)
	 * @param length number of characters to remove
	 */
	static void remove_at( text& s, natural pos, natural length );

	/**
	 * check if text s starts with subtext st_str
	 * @param s main text
	 * @param st_str subtext that might start s
	 * @return true if st_str starts s, false otherwise
	 */
	static bool starts_with( char *s, char *st_str, bool remove_flag = false );
	static bool starts_with( text& s, text& st_str, bool remove_flag = false );
	static bool starts_with( text& s, char *st_str );
	static bool starts_with( text& s, const char *st_str );

	/**
	 * check if text s ends with st_str and remove it if remove_flag is set
	 * @param s text to check
	 * @param st_str text to look for
	 * @param remove_flag indicates if we should remove st_str if it is found
	 */
	static bool ends_with( text& s, text& st_str, bool remove_flag = false );
	static bool ends_with( text& s, char *st_str, bool remove_flag = false );
	static bool ends_with( text& s, const char *st_str, bool remove_flag = false );



};

} // end of namespace essential

} // end of namespace ez

#endif /* ESSENTIAL_TEXT_UTILS_H_ */
