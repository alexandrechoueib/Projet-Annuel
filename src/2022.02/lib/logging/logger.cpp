/*
 * logger.cpp
 *
 *  Created on: Jul 28, 2013
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "logging/logger.h"

using namespace ez::logging;

const integer Logger::CLOSED = 0;
const integer Logger::LEVEL_1 = 1;
const integer Logger::LEVEL_2 = 2;
const integer Logger::LEVEL_3 = 3;
const integer Logger::LEVEL_4 = 4;
const integer Logger::LEVEL_5 = 5;
const integer Logger::MAX_LEVEL = 15;


Logger::Logger( text name, std::ostream *output)  : BaseLogger() {

	_name = name;
	_output = output;
	_record_level = 1;
	_verbose_level = 5;
	_header_format = "%n %l %d %t$ ";

}


Logger::~Logger() {

}


void Logger::set_record_level( integer lvl ) {

	if ((lvl < CLOSED) || (lvl > MAX_LEVEL)) {
		notify("bad value, record level should be between " << CLOSED << " and " << MAX_LEVEL );
	}
	_record_level = lvl;
	
}


void Logger::inc_record_level() {

	++_record_level;

}


void Logger::dec_record_level() {

	--_record_level;

}


void Logger::set_verbose_level( integer lvl ) {

	if ((lvl < CLOSED) || (lvl > MAX_LEVEL)) {
		notify("bad value, record level should be between " << CLOSED << " and " << MAX_LEVEL );
	}
	_verbose_level = lvl;
	
}


void Logger::set_header( const char *s ) {
	
	_header_format = s;

}


void Logger::print_header() {

	time_t now;
	time( &now );
	struct tm *my_time = localtime( &now );
	char tmp[30];

	size_t i = 0;
	while (i < _header_format.size()) {
		char c = _header_format[ i ];
		if (c == '%') {
			++i;
			switch( _header_format[ i ] ) {
			case 'd':
				strftime( tmp, 29, "%Y-%m-%d", my_time );
				(*_output) << tmp;
				break;
				
			case 't':
				strftime(tmp, 29, "%H:%M:%S", my_time);
				(*_output) << tmp;
				break;
				
			case 'n':
				(*_output) << _name;
				break;
				
			case 'l':
				(*_output) << "(level " << _record_level << ")";
				break;
				
			default:
				break;
			}
			
		} else {
			(*_output) << c;
		}
		++i;
	}

}


void Logger::print_ln() {

	print('\n');

}


void Logger::print( char v ) {

	if (_record_level > _verbose_level) return ;
	if (v == '\n') {
		print_header();
		(*_output) << _input.str() << std::endl;
		_output->flush();
		_input.str( "" );
		
	} else {
		_input << v;
	}
	
}


void Logger::print( integer v ) {
	
	if (_record_level > _verbose_level) return ;
	_input << v;
	
}


void Logger::print( natural v ) {
	
	if (_record_level > _verbose_level) return ;
	_input << v;

}


void Logger::print( const char *v ) {
	
	if (_record_level > _verbose_level) return ;
	while (*v != '\0') {
		print( *v );
		++v;
	}
	
}


void Logger::print( string v ) {

	if (_record_level > _verbose_level) return ;
	for (size_t i = 0; i < v.size(); ++i) {
		print( v[ i ] );
	}
}


void Logger::print( real v ) {

	if (_record_level > _verbose_level) return ;
	_input << v;

}


void Logger::print( void *v ) {
	
	if (_record_level > _verbose_level) return ;
	_input << hex << v << dec;
	
}


void Logger::print( const _Setrecordlevel& _lvl ) {

	set_record_level( _lvl._M_n );

}


void Logger::print( const _Setverboselevel& _lvl ) {

	set_verbose_level(_lvl._M_n); 
}


void Logger::print( const ez::essential::CPUTimer& obj ) {

	if (_record_level > _verbose_level) return ;
	_input << const_cast<CPUTimer&>( obj ) << std::endl;

}


void Logger::print( const ez::essential::Range& obj ) {

	if (_record_level > _verbose_level) return ;
	_input << const_cast<Range&>( obj ) << std::endl;

}

// ==================================================================
// CONSOLE
// ==================================================================

ConsoleLogger::ConsoleLogger( string name, ostream *out )  
	: Logger(name, out){

}


ConsoleLogger::~ConsoleLogger() {

}


ConsoleLogger& ConsoleLogger::operator<<( text& s ) {

	print(s);
	return *this;

}

// ==================================================================
// MEMORY
// ==================================================================

MemoryLogger::MemoryLogger( text name ) 
	: Logger( name, &_output_stream ){

}


MemoryLogger::~MemoryLogger() {

}


text MemoryLogger::get_contents() {

	return _output_stream.str();

}


void MemoryLogger::clear() {

	_output_stream.str( "" );

}


MemoryLogger& MemoryLogger::operator<<( text& s ) {

	print( s );
	return *this;
	
}

// ==================================================================
// FILE
// ==================================================================

const int FileLogger::TRUNCATE = 1;


FileLogger::FileLogger( text name, text fn, integer options ) 
	: Logger(name, NULL) {

	_options = options;

	std::_Ios_Openmode mode = std::_Ios_Openmode(0);
	if ((options & FileLogger::TRUNCATE) != 0) {
		mode |= ios::trunc;
	} else {
		mode |= ios::app;
	}
	
	_stream.open( fn.c_str(), mode );

	if (_stream.fail()) {
		notify( "could not open file: " << fn );
	}
	
	_output = &_stream;

}


FileLogger::~FileLogger() {

	_stream.flush();
	_stream.close();

}
