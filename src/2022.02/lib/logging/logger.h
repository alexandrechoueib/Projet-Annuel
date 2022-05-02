/*
 * logger.h
 *
 *  Created on: Jul 28, 2013
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

#ifndef LOGGING_LOGGER_H_
#define LOGGING_LOGGER_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
#include <stdlib.h>
#include <time.h>

#include "essential/cpu_timer.h"
#include "essential/exception.h"
#include "essential/range.h"
#include "essential/types.h"
#include "objects/object.h"

using namespace std;
using namespace ez::essential;

namespace ez {

namespace logging {

class Logger;
class LogManager;

/**
 * Base class for Logger used to define kind of iomanip operations
 * for Loggers
 */
class BaseLogger {
public:
	/**
	 * default constructor
	 */
	BaseLogger() {

	}


	/**
	 * destructor
	 */
	virtual ~BaseLogger() {

	}


	/**
	 * virtual function used to set record level of information sent to
	 * Logger
	 * @param n level of information
	 */
	virtual void set_record_level( integer n ) = 0;


	/**
	 * virtual function used to set verbose level of information sent to
	 * Logger
	 * @param n level of information
	 */
	virtual void set_verbose_level( integer n ) = 0;


	/**
	 * virtual function used to print new line
	 */
	virtual void print_ln() = 0;

};

// Logger structure used to set level of Logger
struct _Setrecordlevel { integer _M_n; };

/**
 * function used to set record level of a Logger
 */
inline _Setrecordlevel
setrecord( integer __n ) {
	_Setrecordlevel __x;
	__x._M_n = __n;
	return __x;
}

// Logger structure used to set level of Logger
struct _Setverboselevel { integer _M_n; };

/**
 * function used to set record level of a Logger
 */
inline _Setverboselevel
setverbose( integer __n ) {
	_Setverboselevel __x;
	__x._M_n = __n;
	return __x;
}


/**
 * template function used to set record level information
 */
template<class T>
inline Logger&  operator<<( BaseLogger& l, _Setrecordlevel __f ) {

	l.set_record_level( __f._M_n );
	return dynamic_cast<Logger&>( l );
	
}


/**
 * template function used to set verbose level information
 */
template<class T>
inline Logger&  operator<<( BaseLogger& l, _Setverboselevel __f ) {

	l.set_verbose_level( __f._M_n );
	return dynamic_cast<Logger&>( l );
	
}


/**
 * base class for loggers which is composed of:
 * <ul>
 * <li> a name which serves as an identifier to retrieve
 * the logger from the LoggerManager</li>
 * <li> a pointer to an output stream</li>
 * </ul>
 * The Logger needs to be attached to the LoggerManager (@see LoggerManager).
 * Use << setlevel(int) to fix the level of information recorded.
 */
class Logger : public BaseLogger {
public:
	/**
	 * information will not be recorded
	 */
	static const integer CLOSED;
	static const integer LEVEL_1;
	static const integer LEVEL_2;
	static const integer LEVEL_3;
	static const integer LEVEL_4;
	static const integer LEVEL_5;
	
	/**
	 * maximum level for information recording
	 */
	static const integer MAX_LEVEL;


	/**
	 * constructor to build new Logger
	 * @param name identifier of the logger
	 * @param out pointer to output stream
	 */
	Logger(text name, std::ostream *out);


	/**
	 * destructor
	 */
	virtual ~Logger();


	/**
	 * set logging level of information sent to logger
	 */
	void set_record_level( integer lvl );


	/**
	 * increment record level
	 */
	void inc_record_level();


	/**
	 * decrement record level
	 */
	void dec_record_level();


	/**
	 * set level of information displayed, if record level is greater
	 * than verbose level, then information is not shown
	 */
	void set_verbose_level( integer lvl );

	/**
	 * set header displayed at the beginning of each line by using the
	 * following format to display:
	 * <ul>
	 * <li> %<code>d</code> for day</li>
	 * <li> %<code>t</code> for time</li>
	 * <li> %<code>n</code> logger name</li>
	 * <li> %<code>m</code> for message</li>
	 * <li> %<code>l</code> for record level</li>
	 * </ul>
	 */
	void set_header( const char *s );


	/**
	 * print new line
	 */
	void print_ln();


	/**
	 * print character
	 */
	void print( char v );


	/**
	 * print integer
	 */
	void print( integer v );


	/**
	 * print unsigned int
	 */
	void print( natural v );


	/**
	 * print pointer to char
	 */
	void print( const char *s );


	/**
	 * print string
	 */
	void print( string s );


	/**
	 * print float/double
	 */
	void print( real f );


	/**
	 * print pointer
	 */
	void print( void *p );


	/**
	 * set record level (use like iomanip)
	 * @param lvl _Setlevel structure
	 */
	void print( const _Setrecordlevel& lvl );


	/**
	 * set verbose level (use like iomanip)
	 * @param lvl _Setlevel structure
	 */
	void print( const _Setverboselevel& lvl );


	/**
	 * print CPUTimer on Logger
	 */
	void print( const ez::essential::CPUTimer& obj );


	/**
	 * Print Range on Logger
	 */
	void print( const ez::essential::Range& obj );
	
	

	typedef std::ostream& (*ManipFn)( std::ostream& );
	typedef std::ios_base& (*FlagsFn)( std::ios_base& );


	// endl, flush, setw, setfill, etc.
	Logger& operator<<( ManipFn manip ) {
	
		manip( _input );
		
		if (manip == static_cast<ManipFn>( std::flush ) || manip == static_cast<ManipFn>( std::endl )) {
			(*_output) << _input.str();
			(*_output).flush();
			_input.str( "" );
			
		}
		
		return *this;
		
	}


	// setiosflags, resetiosflags
	Logger& operator<<( FlagsFn manip ) {
	
		manip( _input );
		
		return *this;
		
	}


	/**
	 * template function used to display objects.
	 * The object must have a void display(ostream& out) method.
	 */
	template<class T>
	void print_object( T *obj ) {
	
		if (_record_level > _verbose_level) return ;
		obj->display( _input );
		print( '\n' );
		
	}


	/**
	 * return pointer to output stream
	 * @return pointer to output stream
	 */
	std::ostream *output_stream() {
	
		return _output;
		
	}


	/**
	 * overloading of output operator
	 */
	template<class T>
	Logger& operator<<( const T& a ) {
	
		print(a);
		return *this;
		
	}


	friend class LoggerManager;

protected:
	/**
	 * pointer to an output stream for example &std::cout
	 */
	std::ostream *_output;
	/**
	 * input stream which stores information until a new line
	 * character is found. The contents of input is then sent
	 * to the output stream
	 */
	std::ostringstream _input;
	/**
	 * identifier of Logger
	 */
	std::string _name;
	/**
	 * string used as header displayed at the beginnig of each line
	 */
	std::string _header_format;

	/**
	 * level of information that is sent to logger
	 * if loggin_level > display level, information
	 * won't be recorded
	 */
	integer _record_level;
	/**
	 * level of information displayed
	 */
	integer _verbose_level;


	/**
	 * display header at the beginning of the line
	 */
	void print_header();

};

/**
 * Logger that displays information on stdout or stderr
 */
class ConsoleLogger : public Logger {
public:
	/**
	 * constructor
	 * @param name identifier of the logger
	 * @param out output stream
	 */
	ConsoleLogger( text name, std::ostream *out=&std::cerr );


	/**
	 * destructor
	 */
	~ConsoleLogger();


	ConsoleLogger& operator<<( text& s );

};


/**
 * Logger for which information is kept in memory
 */
class MemoryLogger : public Logger {
public:
	/**
	 * constructor
	 * @param name identifier of the logger
	 */
	MemoryLogger( text name );


	/**
	 * destructor
	 */
	~MemoryLogger();


	/**
	 * get contents
	 */
	text get_contents();

	/**
	 * clear information stored
	 */
	void clear();
	

	MemoryLogger& operator<<( text& s );

protected:
	/**
	 * ostringstream used to store information
	 */
	std::ostringstream _output_stream;
	
};


/**
 * Logger that uses a file to store information
 */
class FileLogger : public Logger {
public:
	static const int TRUNCATE;

	/**
	 * constructor
	 * @param name identifier of logger
	 * @param fn file name
	 * @param opt options, for example TRUNCATE will empty the file
	 */
	FileLogger( text name, text fn, integer options = 0 );
	
	
	/**
	 * destructor
	 */
	~FileLogger();

protected:
	std::ofstream _stream;
	integer _options;
	
};

} // end of namespace logging

} // end of namespace ez

#endif /* LOGGING_LOGGER_H_ */
