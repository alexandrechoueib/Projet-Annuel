/*
 * log_manager.cpp
 *
 *  Created on: Jul 28, 2013
 * Modified on: Feb, 2022
 *      Author: richer
 */

#include "logging/logger_manager.h"

using namespace ez::logging;

LoggerManager *LoggerManager::_instance = nullptr;

Logger *LoggerManager::_root_logger = nullptr;


LoggerManager::LoggerManager() {

	// create a console root logger
	_root_logger = new ConsoleLogger( "root_logger", &std::cerr );
	_root_logger->set_verbose_level( Logger::CLOSED );

}


LoggerManager& LoggerManager::instance() {

	if (_instance == NULL) {
		_instance = new LoggerManager;
	}

	return *_instance;

}


LoggerManager::~LoggerManager() {

	map<text, Logger *>::iterator iter;
	
	for (iter = _loggers.begin(); iter != _loggers.end(); ++iter) {
		delete (*iter).second;
	}
	
	_loggers.clear();
	
}


void LoggerManager::attach( Logger *l ) {

	map<text, Logger *>::iterator iter = _loggers.find( l->_name );
	
	if (iter != _loggers.end()) {
		notify( "Logger of name [" << l->_name << "] already exists" );
	}
	
	_loggers[ l->_name ] = l;
	
}


Logger *LoggerManager::detach( text name ) {

	map<string, Logger *>::iterator iter = _loggers.find( name );
	
	if (iter == _loggers.end()) {
		notify( "Logger of name [" << name << "] does not exist" );
	}
	
	Logger *logger = (*iter).second;
	_loggers.erase( iter );
	
	return logger;
	
}


Logger& LoggerManager::get( text name, text or_create ) {

	map<string, Logger *>::iterator iter = _loggers.find( name );
	
	if (iter == _loggers.end()) {
		if (or_create.size() == 0) {
			notify( "Logger of name [" << name << "] does not exists or has not been recorded"
				<< " to the LogManager" );
		} else {
			return create( or_create );
		}
	}
	
	return *_loggers[ name ];
	
}

Logger& LoggerManager::root_logger() {

	return *_root_logger;

}


void LoggerManager::explode( text& s, std::vector<text>& v ) {

	string delim = ",";
	// Skip delimiters at beginning.
	string::size_type lastPos = s.find_first_not_of( delim, 0 );
	// Find first "non-delimiter".
	string::size_type pos = s.find_first_of( delim, lastPos );

	while (string::npos != pos || string::npos != lastPos) {
		// Found a token, add it to the vector.
		v.push_back( s.substr( lastPos, pos - lastPos ) );
		// Skip delimiters.  Note the "not_of"
		lastPos = s.find_first_not_of( delim, pos );
		// Find next "non-delimiter"
		pos = s.find_first_of( delim, lastPos );
	}
	
}

Logger& LoggerManager::create( text s ) {

	vector<text> parameters;

	explode( s, parameters );
	
	if (parameters.size() == 0) {
		notify( "LoggerManager:create:no parameter provided for logger creation" );
	}

	Logger *log = nullptr;
	text name = "root_logger";

	if (parameters[ 0 ] == "console") {
		if (parameters.size() == 1) {
			log = new ConsoleLogger( name );
			
		} else {
			if (parameters.size() == 2) {
				name = parameters[ 1 ];
				if (parameters[ 1 ] == "stdout") {
					log = new ConsoleLogger( name, &std::cout );
				} else if (parameters[1] == "stderr") {
					log = new ConsoleLogger( name, &std::cerr );
				} else {
					notify( "LoggerManager:create:for console logger: stdout or stderr are expected" );
				}
			} else {
				notify( "LoggerManager:create:bad number of parameters for console logger" );
			}

		}
		
	} else if (parameters[ 0 ] == "file") {
		if (parameters.size() == 1) {
			notify( "LoggerManager:create:missing file name in definition of file logger" );
		} else if (parameters.size() == 2) {
			name = parameters[ 1 ];
			log = new FileLogger( name, parameters[ 1 ] );
		} else if (parameters.size() == 3) {
			name = parameters[ 1 ];
			if (parameters[ 2 ] == "truncate") {
				log = new FileLogger( name, parameters[ 1 ], FileLogger::TRUNCATE );
			} else {
				notify( "LoggerManager:create:truncate expected in definition of file logger" );
			}
		}

	} else {
		notify( "LoggerManager:create:type of logger is unknown : '" << parameters[ 0 ] << "'" );
	}

	delete _root_logger;
	_root_logger = log;
	return *log;
	
}

