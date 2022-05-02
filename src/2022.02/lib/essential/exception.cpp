/*
 * exception.cpp
 *
 *  Created on: Apr 7, 2017
 *      Author: Jean-Michel Richer
 */

#include "essential/exception.h"

using namespace ez::essential;

/**
 * Stream where causes of exceptions are described
 */
std::ostringstream cexc;

/**
 * Stack of exceptions
 */
std::stack<std::string> stack_of_exceptions;


Exception::Exception( const char *in_file, int at_line ) : std::exception() {

	_in_file = in_file;
	_at_line = at_line;
	
	std::ostringstream oss;
	oss << "in " << _in_file << " at line " << _at_line << ": " << cexc.str();
	stack_of_exceptions.push( oss.str() );
	// clear stream of exceptions
	cexc.str( "" );
	
}


Exception::~Exception() throw() {

	cexc.str("");

}

const char *Exception::what() const throw() {
	//std::cerr << "EXCEPTION" << std::endl;
	//std::cerr << "cexc=" << cexc.str()<< std::endl;
	//std::cerr << "cexr=" << cexr.str()<< std::endl;
	//std::cerr << "in file = " << m_in_file << std::endl;
	//std::cerr << "at line = " << m_at_line << std::endl;

	std::ostringstream oss;

	oss << "\t" << stack_of_exceptions.top() << std::endl;

	std::string *s_tmp = new std::string( oss.str() );

	return const_cast<char *>( s_tmp->c_str() );

}

void Exception::clear() {

	while (!stack_of_exceptions.empty()) stack_of_exceptions.pop();
	
}

void Exception::print_stack_trace( std::ostream& out ) {

	if (stack_of_exceptions.empty()) return ;
	
	size_t level = stack_of_exceptions.size();
	
	while (!stack_of_exceptions.empty()) {
		out << level-- << ": " << stack_of_exceptions.top() << std::endl;
		stack_of_exceptions.pop();
	}

}
