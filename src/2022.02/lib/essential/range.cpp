#include "range.h"
using namespace ez::essential;

/**
 @WHAT
   Check if value @param:v is inside authorized interval;
   i.e from -(2^31-1) to (2^31-1)
   
 */
void __check_limit(integer v) {
	i64 v64 = static_cast<i64>( v );
	if ((static_cast<i64>(RANGE_MIN) <= v64) && (static_cast<i64>(RANGE_MIN) <= v64)) {
		return ;
	} else {
		notify( "Range value " << v << " is not in the authorized interval ["
			<< RANGE_MIN << ".." << RANGE_MAX << "]" );
	}
}


Range::Range() {

	_fst_value = RANGE_NULL;
	_lst_value = RANGE_NULL;
	_increment = 1;
	_size = 0;
	
}
	
Range::Range( int first_value, int last_value, int increment ) {

	__check_limit( first_value );
	__check_limit( last_value );
	
	_fst_value = first_value;
	_lst_value = last_value;
	_increment = increment;
	
	if (_lst_value < _fst_value) {
		ensure(_increment < 0);
	}	
	
	__compute_size();

}


void Range::values( std::vector<int>& values ) {

	if (_increment >= 1) {
		int v = _fst_value;
		while (v <= _lst_value) {
			values.push_back( v );
			v += _increment;
		}
		
	} else if (_increment <= -1) {
		int v = _fst_value;
		while (v >= _lst_value) {
			values.push_back( v );
			v += _increment;
		}
		
	}	
 
}

bool Range::is_null() {
	return (_fst_value == RANGE_NULL) && (_lst_value == RANGE_NULL);
}

