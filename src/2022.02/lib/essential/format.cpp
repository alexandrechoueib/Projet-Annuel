/*
 * format.cpp
 *
 *  Created on: Apr 24, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "essential/format.h"

using namespace ez::essential;

text Format::_format(text s, integer distance) {

	integer l = static_cast<integer>( s.length() );

	text res;

	integer k = l % distance;
	for (integer i = 0; i<l; ++i) {
		if (((i % distance) == k) && (i != 0)) {
			res += '_';
		}
		res += s[i];
	}

	return res;
	
}


text Format::_to_base( natural n, NumericBase& base ) {

	std::string s;

	natural divider = base._digits;

	while (n >= divider) {
		s += base._output_symbols[ ( n % divider ) ];
		n /= divider;
	}
	
	s += base._output_symbols[ ( n % divider ) ];
	reverse( s.begin(), s.end() );
	
	return s;

}


text Format::dec( natural n ) {

	NumericBase& base = NumericBaseManager::instance().by_id( Base::DECIMAL );

	return _format( _to_base( n, base ), 3 );

}


text Format::bin( natural n, natural size_in_bits ) {

	if (size_in_bits == 8) {
		n &= 0xFF;
	} else if (size_in_bits == 16) {
		n &= 0xFFFF;
	} else if (size_in_bits == 32) {
		n &= 0xFFFFFFFF;
	}
	
	NumericBase& base = NumericBaseManager::instance().by_id( Base::BINARY );

	return _format( _to_base( n, base ), 4 ) + "_b";

}


text Format::oct( natural n ) {

	NumericBase& base = NumericBaseManager::instance().by_id( Base::OCTAL );

	return _format( _to_base( n, base ), 3 ) + "_o";

}


text Format::hex( long_natural n ) {

	NumericBase& base = NumericBaseManager::instance().by_id( Base::HEXADECIMAL );

	return _format( _to_base( n, base ), 2 ) + "_h";

}


text Format::left( text s, integer width, character fill_char ) {

	ensure(width >= 0);

	text res = s;

	if (static_cast<integer>( res.length() ) < width) {
		std::fill_n( back_inserter( res ), width - res.length(), fill_char);
	}
	
	return res;
	
}


text Format::center( text s, integer width, character fill_char ) {

	ensure(width >= 0);

	text res = s;

	if (static_cast<integer>( res.length() ) < width) {
		std::fill_n( back_inserter( res ), (width - res.length()) / 2, fill_char );
		std::fill_n( inserter( res, res.begin() ), width -res.length(), fill_char );
	}

	return res;

}


text Format::right( text s, integer width, character fill_char ) {

	ensure(width >= 0);

	text res = s;

	if (static_cast<integer>( res.length() ) < width) {
		std::fill_n( inserter( res, res.begin() ), width - res.length(), fill_char );
	}
	
	return res;
	
}


text Format::fp( f32 n, natural size, natural decimals ) {
	text res;
	std::ostringstream oss;

	oss << std::fixed;

	if (size > 0) {
		oss.width( size );
	}
	
	if (decimals > 0) {
		oss << std::setprecision( decimals );
	}
	
	oss << n;
	res = oss.str();
	return res;
	
}

text Format::fp( f64 n, natural size, natural decimals ) {

	text res;
	std::ostringstream oss;

	oss << std::fixed;

	if (size > 0) {
		oss.width( size );
	}
	
	if (decimals > 0) {
		oss << std::setprecision( decimals );
	}
	
	oss << n;
	res = oss.str();
	return res;
	
}
