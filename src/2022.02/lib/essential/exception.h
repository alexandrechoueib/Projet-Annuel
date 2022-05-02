/*
 * exception.h
 *
 *  Created on: Apr 7, 2017
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

#ifndef ESSENTIAL_EXCEPTION_H_
#define ESSENTIAL_EXCEPTION_H_

#include <stack>
#include <stdexcept>
#include <string>
#include <iostream>
#include <sstream>

/**
 * default stream used to describe the cause of the exception
 */
extern std::ostringstream cexc;

namespace ez {

namespace essential {

/**
 * raise exception and notify cause
 */
#define notify(reason) \
	cexc << reason; \
	throw ez::essential::Exception(__FILE__, __LINE__)



/**
 @CLASS
   Exception for the EZ Library. We are using the cexc stream 
   to provide some message about the cause of the error and 
   we record the file and line where it occured.
   
 */
class Exception : public std::exception {
public:

	// source file where the exception occurred
	const char *_in_file;

	// line in the source file
	int _at_line;

public:
	/**
	 @WHAT
	   Default constructor
	   
	 @PARAMETERS
	   @param:in_file name of file where the expcetion occurred
	   @param:at_line line in the file
	     
	 */
	Exception( const char *in_file, int at_line );

	/**
	 @WHAT
	   Destructor
	 
	 */
	~Exception() throw();

	/**
	 @WHAT
	   Return error message to print
	   
	 */
	const char *what() const throw();

	/**
	 @WHAT
	   Print trace of exceptions
	   
	 */
	void print_stack_trace( std::ostream& out );

	/**
	 @WHAT
	   Clear trace
	   
	 */
	static void clear();
};

} // end of namespace essential

} // end of namespace ez

#endif /* ESSENTIAL_EXCEPTION_H_ */
