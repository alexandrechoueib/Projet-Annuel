/*
 * format.h
 *
 *  Created on: Apr 21, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

/*
    EZLib version 2022.02
    Copyright (C) 2019-2022 Jean-Michel Richer

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

#ifndef ESSENTIAL_FORMAT_H_
#define ESSENTIAL_FORMAT_H_

#include <iomanip>

#include "essential/ensure.h"
#include "essential/numeric_base.h"
#include "essential/types.h"
namespace ez {

namespace essential {

#define ez_to_bin(x) \
	ez::essential::Format::bin(static_cast<ez::essential::long_natural>(x))

/**
 @CLASS
   Class used to pretty print data.
   
 */
class Format {
protected:

	/**
	 @WHAT
	   Convert natural number n to string in a given base
	   
	 */
	static text _to_base( natural n, NumericBase& base );
	
	/**
	 @WHAT
	   Modify input string so that every distance characters
	   we add a separator '_' in order to have a better
	   visualization of a number transformed as text
	   
	 @EXAMPLE
	 	text s = "110010100";
	 	text t = _format(s, 4);
	 	// then t = "1_1001_0100";
	 
	 */
	static text _format( text s, integer distance );

public:
	/**
	 @WHAT
	   Print a string on the left of a given width and fill with spaces
	  
	 @PARAMTERS 
	  @param s string to print
	  @param width
	  @param fill_char character to use to fill rest of the width
	  
	 */
	static text left( text s, integer width, character fill_char=' ' );

	/**
	 @WHAT
	   Print a string on the center of a given width and fill with spaces

	 @PARAMTERS 
	  @param s string to print
	  @param width
	  @param fill_char character to use to fill rest of the width
	   
	 */
	static text center( text s, integer width, character fill_char=' ' );

	/**
	 @WHAT
	  Print a string on the right of a given width and fill with spaces

	 @PARAMTERS 
	  @param s string to print
	  @param width
	  @param fill_char character to use to fill rest of the width

	 */
	static text right( text s, integer width, character fill_char=' ' );


	/**
	 @WHAT
	   Convert natural value to decimal format
	   
	 @PARAMETERS
	   @param n value to convert
	     
	 */
	static text dec( natural n );
	
	/**
	 @WHAT 
	   Convert natural number to binary format
	   
	 @PARAMETERS
	   @param n value to convert
	   @param size_in_bits size of the representation  
	   
	 */
	static text bin( natural n, natural size_in_bits=32 );

	/**
	 @PARAM
	  Convert natural number to octal format
	  
	 @PARAMETERS
	  @param n value to convert
	  
	 */
	static text oct( natural n );

	/**
	 @PARAM
	  Convert natural number to hexadecimal format
	  
	 @PARAMETERS
	  @param n value to convert
	  
	 */
	static text hex( long_natural n );

	/**
	 @WHAT
	   convert simple precision floating point value with given 
	   number of decimals and size if greater than 0
	   
	 @PARAMETERS
	   @param n value to convert
	   @param size number of characters used to represent number
	   @param decimals  number of decimals
	 */  
	static text fp( f32 n, natural size=0, natural decimals=0 );

	/**
	 @WHAT
	   convert double precision floating point value with given 
	   number of decimals and size if greater than 0
	   
	 @PARAMETERS
	   @param n value to convert
	   @param size number of characters used to represent number
	   @param decimals  number of decimals
	 */  
	static text fp( f64 n, natural size=0, natural decimals=0 );
	
};

} // end of namespace essential

} // end of namespace ez

#endif /* ESSENTIAL_FORMAT_H_ */
