/*
 * text.h
 *
 *  Created on: Apr 11, 2017
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
#ifndef OBJECTS_TEXT_H_
#define OBJECTS_TEXT_H_

#include <memory>
#include <string>
#include <cstring>

#include "objects/object.h"
using namespace std;
#include "essential/text_utils.h"
#include "objects/object.h"

namespace ez {

namespace objects {

class Text : public Object {
public:
	typedef Text self;

	text m_value;

	Text() : Object() {

	}

	Text(text x) : Object() {
		m_value = x;
	}

	Text(const char *x) : Object() {
		m_value = x;
	}

	Text(const self& obj) : Object() {
		m_value = obj.m_value;
	}

	self& operator=(const self& obj) {
		if (&obj != this) {
			m_value = obj.m_value;
		}
		return *this;
	}

	std::ostream& print(std::ostream& stream);


	/**
	 * equality between objects
	 */
	integer compare(const Object& y) {
		Text& y_obj = *dynamic_cast<Text *>(&const_cast<Object&>(y));
		if (m_value == y_obj.m_value) return 0;
		return (m_value < y_obj.m_value) ? -1 : +1;
	}

	Object *clone() {
		return new Text(m_value);
	}

	/**
	 * remove all characters that correspond to any character of pattern
	 * at the end of the text
	 */
	void trim_right(const text& pattern=TEXT_BLANKS);

	/**
	 * remove all characters that correspond to any character of pattern
	 * at beginning of the text
	 */
	void trim_left(const text& pattern=TEXT_BLANKS);

	/**
	 * remove all characters that correspond to any character of pattern
	 * at beginning and end of the text
	 */
	void trim(const text& pattern=TEXT_BLANKS);

	/**
	 * erase all characters that correspond to any character of pattern
	 */
	void erase(const text& pattern=TEXT_BLANKS);

	/**
	 * remove all spaces characters
	 */
	void remove_spaces();

	/**
	 * replace all occurrences of source character by replacement character
	 */
	void replace_char(char src, char repl);

	/**
	 * convert text to lowercase
	 */
	void lower();

	/**
	 * convert text to uppercase
	 */
	void upper();

	/**
	 * split text in subtexts when separated by any character of delimiters
	 * the subtexts are put in a vector
	 * @param s text to explode
	 * @param v output subtexts
	 * @param delim characters used as delimiters
	 */
	void explode(vector<text>& v, const text& delim=TEXT_BLANKS);

	/**
	 * join texts of a vector into a single text where delimiters separate
	 * each subtext
	 * @param s text used as a result
	 * @param v vector of texts to implode
	 *
	 */
	void implode(vector<text>& v, const text& delim=TEXT_BLANK);
	void implode(vector<char>& v, const text& delim=TEXT_BLANK);

	/**
	 * return position of st_str in s or text::npos if not found
	 * @param s text that will be searched
	 * @param st_str subtext searched in s
	 * @return position of st_str in s, pr text::npos
	 */
	size_t position_of(text& st_str);

	/**
	 * return position of st_str in s or text::npos if not found
	 * @param s text that will be searched
	 * @param st_str subtext searched in s
	 * @return position of st_str in s, pr text::npos
	 */
	size_t position_of(char *st_str);

	/**
	 *
	 */
	void remove_at(natural pos, natural length);

	/**
	 * check if text s starts with subtext st_str
	 * @param s main text
	 * @param st_str subtext that might start s
	 * @return true if st_str starts s, false otherwise
	 */
	bool starts_with(text& st_str);
	bool starts_with(char *st_str);
	bool starts_with(Text& st_str);
	bool ends_with(text& st_str, bool rm_flag=false);
	bool ends_with(char *st_str, bool rm_flag=false);
	bool ends_with(Text& st_str, bool rm_flag=false);

	static string _text_(boolean b);
	static string _text_(character c);
	static string _text_(integer i);
	static string _text_(real r);


};

}

}

#endif /* OBJECTS_TEXT_H_ */
