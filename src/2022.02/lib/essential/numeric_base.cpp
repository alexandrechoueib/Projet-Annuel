/*
 * numeric_base.cpp
 *
 *  Created on: Apr 22, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "essential/numeric_base.h"

#include "essential/text_utils.h"

using namespace ez::essential;


NumericBaseManager *NumericBaseManager::_instance = nullptr;


NumericBaseManager& NumericBaseManager::instance() {

	if (_instance == nullptr) {
		_instance = new NumericBaseManager();
	}

	return *_instance;
	
}


NumericBaseManager::NumericBaseManager() {

	NumericBase *b_02 = new NumericBase( Base::BINARY, 
		"binary", 'b', 2, "+-01_", "01" );
		
	NumericBase *b_08 = new NumericBase( Base::OCTAL, 
		"octal", 'o', 8, "+-01234567", "01234567" );
		
	NumericBase *b_10 = new NumericBase( Base::DECIMAL, 
		"decimal", 'd', 10, "+-0123456789_", "0123456789" );
		
	NumericBase *b_16 = new NumericBase( Base::HEXADECIMAL, 
		"hexadecimal", 'h', 16, "+-0123456789abcdefABCDEF_", 
		"0123456789abcdefABCDEF" );

	_decimal = b_10;

	_bases.push_back( b_02 );
	_bases.push_back( b_08 );
	_bases.push_back( b_10 );
	_bases.push_back( b_16 );

	_id_access[ Base::BINARY ] = b_02;
	_id_access[ Base::OCTAL ] = b_08;
	_id_access[ Base::DECIMAL ] = b_10;
	_id_access[ Base::HEXADECIMAL ] = b_16;

	_name_access[ "binary" ] = b_02;
	_name_access[ "octal" ] = b_08;
	_name_access[ "decimal" ] = b_10;
	_name_access[ "hexadecimal" ] = b_16;

}


NumericBase& NumericBaseManager::by_name( text _name ) {

	std::vector<NumericBase *>::iterator it;

	for (it = _bases.begin(); it != _bases.end(); ++it) {
		if ((*it)->_name == _name) {
			return *( *it );
		}
	}
	
	notify("numeric base of name '" << _name << "' does not exist");
	return *_decimal;
	
}


NumericBase& NumericBaseManager::by_id( Base b ) {

	return *_id_access[ b ];

}


NumericBase& NumericBaseManager::by_suffix( text number ) {

	if (number.size() == 0) {
		return *_decimal;
	}

	char last_char = number[ number.size()-1 ];
	if (!isalpha(last_char)) return *_decimal;

	for (auto b : _bases) {
		if (b->_suffix[ 0 ] == last_char) return *b;
	}
	
	return *_decimal;
	
}


text NumericBase::convert( natural n ) {

	text s;

	natural divider = _digits;

	while (n >= divider) {
		s += _output_symbols[ (n % divider) ];
		n /= divider;
	}
	
	s += _output_symbols[ (n % divider) ];
	reverse( s.begin(), s.end() );
	
	return s + "_" + _suffix;

}

