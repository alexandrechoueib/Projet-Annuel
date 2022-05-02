/*
 * types.cpp
 *
 *  Created on: Apr 18, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "essential/types.h"

using namespace ez::essential;


bool Types::_ends_with_remove( text& s, text sub ) {

	bool result = false;
	if (s.size() < sub.size()) return result;
	natural length = sub.size();
	result = strcmp( &s.c_str()[s.size()-length], sub.c_str() ) == 0;

	if (result) {
		s.erase( s.begin() + s.length() - length, s.end()) ;
	}
	return result;
	
}


NumericBase& Types::_find_base(text& s) {

	s.erase( std::remove(s.begin(), s.end(), '_'), s.end() );

	NumericBase& base = NumericBaseManager::instance().by_suffix( s );
	_ends_with_remove( s, base._suffix );

	text::size_type pos = s.find_first_not_of( base._input_symbols );
	
	if (pos != text::npos) {
		//cout << "allowed chars = " << allowed_chars[base_index] << endl;
		std::ostringstream oss;
		oss << "bad character : '" << s[ pos ] << "' for " << base._name;
		oss << " number";
		throw std::runtime_error( oss.str() );
	}

	return base;
	
}


/**
 *
 */
bool Types::is_integer(text& s) {
	if (s.length() == 0) return false;

	try {
		// work on copy of initial string because it will be modifier
		text s_copy = s;
		NumericBase base = _find_base( s_copy );
		// try to convert number as 64 bits integer
		long_integer value = std::stoll( s_copy.c_str(), 0, base._digits );
		// check that number if the range of 32 bits integers
		if ((value < INT_MIN) || (value > INT_MAX)) throw std::overflow_error("for integer range");
	} catch (std::exception& e) {
		return false;
	}
	
	return true;
	
}


bool Types::is_long_integer( text& s ) {

	if (s.length() == 0) return false;

	try {
		text s_copy = s;
		NumericBase base = _find_base( s_copy );
		std::stoll( s_copy.c_str(), 0, base._digits );
	} catch (...) {
		return false;
	}
	return true;
	
}


bool Types::is_natural( text& s ) {

	if (s.length() == 0) return false;

	long_natural value = 0;
	try {
		text s_copy = s;
		NumericBase base = _find_base( s_copy );
		value = std::stoul( s_copy.c_str(), 0, base._digits );
		if ((value > UINT_MAX)) throw std::overflow_error( "for integer range" );
	} catch (...) {
		//std::cerr << " " << value << std::endl;
		return false;
	}
	
	return true;

}


bool Types::is_long_natural( text& s ) {

	if (s.length() == 0) return false;

	try {
		text s_copy = s;
		NumericBase base = _find_base( s_copy );
		std::stoul( s_copy.c_str(), 0, base._digits );
	} catch (...) {
		return false;
	}
	return true;

}


bool Types::is_real( text& s ) {

	try {
		std::stof( s.c_str() );
	} catch (...) {
		return false;
	}
	return true;
	
}


bool Types::is_character( text& s ) {

	if (s.length() == 0) return false;
	if (s.length() == 1) return true;

	if (s[0]=='\'') {

	} else {

	}
	
	return true;

}


bool Types::is_numeric( text& s ) {

	return Types::is_integer( s ) || Types::is_real( s );
	
}


template<typename T>
bool converter( text& s, T& dst ) {

	std::istringstream iss( s );
	iss >> dst;
	return iss.good();
	
}


integer  Types::to_integer( text& s ) {

	text s_copy = s;
	NumericBase base = _find_base(s_copy);
	return std::stoi( s_copy.c_str(), 0, base._digits );
	
}


natural  Types::to_natural( text& s ) {

	text s_copy = s;
	NumericBase base = _find_base( s_copy );
	//std::cout << "s_copy = " << s_copy << std::endl;
	//std::cout << "s_copy.cvt = " << std::stol(s_copy.c_str(), 0, base.digits) << std::endl;
	return static_cast<natural>( std::stol( s_copy.c_str(), 0, base._digits) );
}


real Types::to_real( text& s ) {

	real value;
	converter<real>( s, value );
	return value;

}
