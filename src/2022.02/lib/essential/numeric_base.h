/*
 * numeric_base.h
 *
 *  Created on: Apr 22, 2017
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

#ifndef ESSENTIAL_NUMERIC_BASE_H_
#define ESSENTIAL_NUMERIC_BASE_H_

#include "essential/exception.h"
#include "essential/scalar_types.h"

namespace ez {

namespace essential {

/**
 * Enumeration to define numeric bases in computer science
 */
enum class Base {
		BINARY      = 1,
		OCTAL       = 2,
		DECIMAL     = 3,
		HEXADECIMAL = 4
};

/**
 * Class used to record information about a numeric base
 * like the name, the number of digits and the symbols
 * used to read or write a number in the base
 */
class NumericBase {
public:
	/**
	 * identifier (@see Base)
	 */
	Base _id;

	/**
	 * number of digits (ex: 2 for binary, 8 for octal, ...)
	 */
	integer _digits;

	/**
	 * name of base (ex: "binary", "octal", ...)
	 */
	text _name;

	/**
	 * symbols allowed when reading the number in the base
	 */
	text _input_symbols;

	/**
	 * symbols used to represent number when printing
	 */
	text _output_symbols;

	/**
	 * suffix used when writing number (ex: "_b" for binary, "_o" for octal
	 * and "_h" for hexadecimal)
	 */
	text _suffix;


	/**
	 @WHAT
	   default constructor
	 */
	NumericBase()
	: _id( Base::BINARY ), _digits( 0 ), _name( "" ),
	  _input_symbols( "" ), _output_symbols( "" ), _suffix( "" ) {

	}

	/**
	 @WHAT
	   Constructor with arguments
	   
	 @PARAMETERS  
	   @param id integer identifier
	   @param name name of the base
	   @param suffix suffix character used to identify number in base (for example 'b' for binary)
	   @param digits number of digits (for example 16 for hexadecimal)
	   @param in_symbols input symbols allowed
	   @param out_symbols output symbols allowed to print a number
	   @see convert()
	 */
	NumericBase( Base id, text name, char suffix, integer digits,
			text in_symbols, text out_symbols )
	: _id( id ), _digits( digits ), _name( name ), _input_symbols( in_symbols ),
	  _output_symbols( out_symbols ) {
	  
		_suffix = suffix;
		
	}

	/**
	 @WHAT
	   Convert natural value into base
	   
	 @PARAMETERS  
	   @param n value to convert
	   @return string that represents the number in the base
	 */
	text convert( natural n );

	/**
	 @WHAT
	   Getter for _name of base
	 */
	text name() {
	
		return _name;
		
	}
	
};

/**
 * Manager for NumericBases that lets you have access to a given base.
 * Note that this class implements the singleton design pattern.
 */
class NumericBaseManager {
private:
	/**
	 * vector used to store pointers to bases
	 */
	std::vector<NumericBase *> _bases;

	/**
	 * fast access using identifier
	 */
	std::map<Base, NumericBase *> _id_access;

	/**
	 * fast access using name
	 */
	std::map<text, NumericBase *> _name_access;

	/**
	 * pointer to decimal base
	 */
	NumericBase *_decimal;

	/**
	 * instance for singleton
	 */
	static NumericBaseManager *_instance;

	/**
	 * Default constructor defined private for singleton
	 */
	NumericBaseManager();

public:
	/**
	 @WHAT
	   Return reference to unique instance of this class
	 */
	static NumericBaseManager& instance();

	/**
	 @WHAT
	   Return reference to base using integer identifier
	   
	 @PARAMETERS  
	   @param:id integer identifier (@see Base)
	   
	 @RETURN  
	    reference to base
	 */
	NumericBase& by_id( Base id );

	/**
	 @WHAT
	   Return reference to base using name
	 
	 @PARAMETERS  
	   @param:_name name of the base (for example "decimal")
	   
	 @RETURN  
	   reference to base or throw exception if name is not found
	 */
	NumericBase& by_name( text _name );

	/**
	 @WHAT 
	   Return reference to base using suffix of base

	 @HOW
	   If there is not suffix i.e. no letter at the end of the number
	  then the base is considered to be decimal.
	   
	 @PARAMETERS  
	   @param number number to analyze as string
	  
	 @RETURN  
	   reference to base
	 */
	NumericBase& by_suffix( text number );
};

} // end of namespace essential

} // end of namespace ez

#endif /* ESSENTIAL_NUMERIC_BASE_H_ */
