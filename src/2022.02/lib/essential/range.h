/*
 * range.h
 *
 *  Created on: Apr 29, 2017
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

#ifndef ESSENTIAL_RANGE_H_
#define ESSENTIAL_RANGE_H_

#include "essential/ensure.h"
#include "essential/exception.h"
#include "essential/language.h"
#include "essential/scalar_types.h"

namespace ez {

namespace essential {

#define RANGE_MIN -2147483647
#define RANGE_MAX +2147483647
#define RANGE_NULL -2147483648


/**
 @CLASS
   This class implements a range that is an increasing
   or decreasing series of integer values where each value 
   is distant from the preceding one from an increment.
  
   A Range can have values in the interval 
   @math:[RANGE_MIN .. RANGE_MAX]@.
  
   A Range is used to define the indices of an Array
   for example (@see:objects/array.h)
  
   Note that there are two types or Range:
   @list
   @item the first one is made of an increment of 1 or -1
   @item the seond one has an increment greater than 1 or 
     smaller than -1
   @endlist
   
   @code
     Range r1(1,10)    // r1 = [1,2,...,9,10]
     Range r2(3,-2,-1) // r2 = [3,2,1,0,-1,-2]
     Range r3(1,12,3)  // r3 = [1,4,7,10]
     Range r4(12,1,-3) // r4 = [12,9,6,3]
   @endcode  
 */
class Range {
private:

	/**
	 * first value of the Range
	 */
	integer _fst_value;
	
	/**
	 * last value of the Range
	 */ 
	integer _lst_value;
	
	/**
	 * increment
	 */
	integer _increment;
	
	/**
	 * number of values in the range
	 */ 
	natural _size;
	
	/**
	 * values of the Range for _increment > 1 or
	 * _increment < -1. 
	 */
	std::vector<int> _values;
	
	
	/**
	 @WHAT
	   	Function that computes the size of a range, i.e. the
	    number of values that compose the range.
	  
	 @HOW
	 	@list
	    @item if the increment is 1 then we have
	      lst_value - fst_value + 1
	    @item if the increment is -1 then we have
	      fst_value - lst_value + 1
	    @item if increment is greater than 1 we populate
	      a vector with the values and return the size
	      of the vector
	    @item if increment is smaller than 1 we populate
	      a vector with the values and return the size
	      of the vector
	    @endlist
	     
	 */
	void __compute_size() {
		_size = 0;
		
		if (_increment == 1) {
			_size = _lst_value - _fst_value + 1;
			
		} else if (_increment == -1) {
			_size = _fst_value - _lst_value + 1;
		
		} else {
		
			if (_increment > 1) {
				int v = _fst_value;
				while (v <= _lst_value) {
					_values.push_back( v );
					v += _increment;
				}
			} else if (_increment < -1) {
				int v = _fst_value;
				while (v >= _lst_value) {
					_values.push_back( v );
					v += _increment;
				}
			
			}	
			_size = _values.size();
						
		} 
		
	}
	
	/**
	 @WHAT
	   Return position of value @param:v inside
	   @member:values using a binary search
	   in an increasing container.
	   The value must be in the @member:values
	   vector. If it is not found an exception
	   is thrown.
	   
	 @PARAMETERS
	   @param:v value to look for
	   
	 @RETURN
	   index of value v in container
	       	
	 */
	natural __bs_find_increasing(int v) {
		natural lo = 0, hi = _values.size()-1, mid;
				
		while (lo <= hi) {
			mid = (lo + hi) / 2;
			if (_values[ mid ] == v) {
				return mid; 
			} else if (v < _values[ mid ]) {
				if (mid == 0) break;
				hi = mid - 1;
			} else {
				lo = mid + 1;
			}
		}
		
		notify("value " << v << " is not in range");
		
	}

	/**
	 @WHAT
	   Return position of value @param:v inside
	   @member:values using a binary search
	   in a decreasing container.
	   The value must be in the @member:values
	   vector. If it is not found an exception
	   is thrown.
	   
	 @PARAMETERS
	   @param:v value to look for
	   
	 @RETURN
	   index of value v in container
	       	
	 */	
	natural __bs_find_decreasing(int v) {
		natural lo = 0, hi = _values.size()-1, mid;
				
		while (lo <= hi) {
			mid = (lo + hi) / 2;
			if (_values[ mid ] == v) {
				return mid; 
			} else if (v < _values[ mid ]) {
				lo = mid + 1;			
			} else {
				if (mid == 0) break;
				hi = mid - 1;
			}
		}
		
		notify("value " << v << " is not in range");
		
	}
	

public:
	typedef Range self;
	
	/**
	 @WHAT 
	 	Default constructor with no values.
	 	
	 */
	Range();
	
	/**
	 @WHAT
	   Constructor with first and last values and increment
	 
	 @PARAMETERS
	   @param:first_value first value of the range which
	   can be positive or negative
	   @param:first_value first value of the range which
	   can be positive or negative
	   @param:increment increment which
	   can be positive or negative
	   
	 */
	Range( int first_value, int last_value, int increment = 1) ;
	
	/**
	 @WHAT
	   Copy constructor
	   
	 @PARAMETERS
	   @param:obj existing Range
	     
	 */
	Range( const Range& obj ) {

		_fst_value = obj._fst_value;
		_lst_value = obj._lst_value;
		_increment = obj._increment;
		_size = obj._size;
		
		// copy values if they are defined
		if (obj._values.size() > 0) {
			copy(obj._values.begin(), obj._values.end(), back_inserter(_values) ); 
		}

	}
	

	/**
	 @WHAT
	   Overloading of assignment operator

	 @PARAMETERS
	   @param:obj existing Range
	   
	 */ 
	self& operator=( const self& obj ) {

		if (&obj != this) {
			_fst_value = obj._fst_value;
			_lst_value = obj._lst_value;
			_increment = obj._increment;
			_size = obj._size;
			
			// copy values if they are defined
			if (obj._values.size() > 0) {
				copy(obj._values.begin(), obj._values.end(), back_inserter(_values) ); 
			}

		}
		
		return *this;
		
	}

	/**
	 */
	bool is_null();
		
	/**
	 @WHAT
	   Return size of the Range, 
	   
	 @HOW
	 	The number of values has already been evaluated 
	 	by the private function @method:__compute_size().
	 	
	 @RETURN
	    Size of the Range	
	   
	 */
	natural size() {
	
		return _size;
		
	}
	
	/**
	 @WHAT
	   Return first value
	   
	 */
	int first_value() {
	
		return _fst_value;
		
	}
	
	/**
	 @WHAT
	   Return last value
	   
	 */
	int last_value() {
	
		return _lst_value;
		
	}
	
	/**
	 @WHAT
	   Return increment
	   
	 */
	int increment() {
		
		return _increment;
		
	}
	
	/**
	 @WHAT
	   Print all values of the Range
	 
	 @PARAMETERS
	   @param:out output stream
	   
	 @RETURN
	   Reference to output stream
	       
	 */
	std::ostream& print_full( std::ostream& out ) {
	
		if ((_increment == 1) || (_increment == -1)) {
			int v = _fst_value;
			while (v <= _lst_value) {
				out << v << " ";
				v += _increment;
			}
			
		} else  {
		
			for (auto v : _values) {
				out << v << " ";
			}
						
		}
		
		return out;
	}

	/**
	 @WHAT
	   Print Range as a triplet first_value:last_value:increment
	 
	 @PARAMETERS
	   @param:out output stream
	   
	 @RETURN
	   Reference to output stream
	       
	 */
	std::ostream& print( std::ostream& out ) {
	
		out << "[" << _fst_value << ":" << _lst_value << ":" << _increment << "]";
			
		return out;
		
	}
	
	/**
	 @WHAT
	   Overloading of output operator

	 @PARAMETERS
	   @param:out output stream
	   @param:obj Range to print
	   
	 */
	friend std::ostream& operator<<( std::ostream& out, Range& obj ) {
	
		return obj.print( out );
		
	}
	
	/**
	 @WHAT
	   Function that returns true if the value @param:v is
	   one of the values of the Range.

	 @PARAMETERS
	   @param:v value to look for

	 @RETURN
	    @style:bold:true@ if value is in the Range,
	    @style:bold:true@ otherwise
	 */
	
	bool contains( int v ) {
		
		if (_increment == 1) {
			return (_fst_value <= v) && (v <= _lst_value);
			
		} else if (_increment == -1) {
			return (_fst_value >= v) && (v >= _lst_value);
			
		} else {
		
			if (_increment > 1) {
				return binary_search(_values.begin(), _values.end(), v);
				
			} else if (_increment < -1) {
				return binary_search(_values.rbegin(), _values.rend(), v);
							
			}
		}
		
		return false;
		
	}
	
	
	/**
	 @WHAT
	    Return index of value in the Range
	 
	 @PARAMETERS
	   @param:v value to look for
	 
	 @RETURN 
	   index of value in range
	     
	 @EXAMPLE
	    if r is a Range(10,20) then value 10 has index 0,
	    value 11 has index 1, ... and value 20 has index 11
	 
	 */
	int to_index( int v ) {
	
		if (_increment == 1) {
			ensure( (v >= _fst_value) && (v <= _lst_value) );
			return v - _fst_value;
			
		} else if (_increment == -1) {
			ensure( (v <= _fst_value) && (v >= _lst_value) );
			return _fst_value - v;
			
		} else {
			
			if (_increment > 1) {
				return (integer) __bs_find_increasing( v );
				
			} else if (_increment < -1) {
				return (integer) __bs_find_decreasing( v );
							
			}
		}
		
		notify("value " << v << " is not in range");
		
		return 0;
		
	}
	
	/**
	 @WHAT
	   Return value of Range at given index
	 	 
	 @PARAMETERS
	   @param:ndx index in the Range
	   
	 @RETURN
	   Value that is at given index
	   
	 @EXAMPLE
	    if r is a Range(10,20) then value 10 is at index 0,
	    value 11 is at index 1, ... and value 20 is at index 11
	  
	 */
	int to_range( int ndx ) {
	
		if (!((ndx >= 0) && (ndx < static_cast<integer>(_size) ) )) {
			notify("index " << ndx << " is not correct. It should be"
			<< " between 0 and " << (static_cast<integer>(_size) -1) );
		}
		
		if (_increment == 1) {
			return _fst_value + ndx;
			
			
		} else if (_increment == -1) {
			return _fst_value - ndx;
			
		} else {
			
			return _values[ ndx ];
		}
	
		notify("index " << ndx << " is not valid");
		return 0;

	}
	
	/**
	 @WHAT
	    Definition of an iterator for the range which is useful for
	    @style:bold:for@ loops
	 */
	class iterator {
	protected:
		// first value
		integer _fst_value;
		// last value
		integer _lst_value;
		// increment
		integer _increment;
		// current value
		integer _curr_value;
		// index related to current value
		integer _index;

	public:
		typedef iterator self_type;
		typedef integer value_type;
		typedef integer& reference;
		typedef integer pointer;
		typedef std::forward_iterator_tag iterator_category;
		typedef int difference_type;

		iterator() : _fst_value(0), _lst_value(0), _increment(1), _curr_value(0), _index(0) { 
		}

		iterator(pointer first_value, pointer last_value, pointer increment) {
			_fst_value = first_value;
			_lst_value = last_value;
			_increment = increment;
			_curr_value = first_value;
			_index = 0;
		}
		

		iterator(const iterator& it) {
			_fst_value = it._fst_value;
			_lst_value = it._lst_value;
			_increment = it._increment;
			_curr_value = it._curr_value;
			_index = it._index;
		}
		
		int index() {
			return _index;
		}

		iterator& operator++() {
			_curr_value += _increment;
			++_index;
			return *this;
		}

		iterator operator++(int junk) {
			iterator tmp(*this);
			_curr_value += _increment;
			++_index;
			return tmp;
		}

		reference operator*() {
			return _curr_value;
		}

		pointer operator->() {
			return _curr_value;
		}

		bool operator==(const iterator& y) {
			if (_increment >= 1) {
				return _curr_value > y._lst_value;
			} else if (_increment <= -1) {
				return _curr_value < y._lst_value;
			} 
			
			return true;
		}

		bool operator!=(const iterator& y) {
			if (_increment >= 1) {
				return _curr_value <= y._lst_value;
			} else if (_increment <= -1) {
				return _curr_value >= y._lst_value;
			}
			
			return true;
		}
	};

	/**
	 @WHAT
	   Definition of begin iterator
	   
	 */
	iterator begin() {
		return iterator(_fst_value, _lst_value, _increment);
	}

	/**
	 @WHAT
	   Definition of end iterator
	   
	 */
	iterator end() {
		return iterator(_fst_value, _lst_value, _increment);
	}
	
	/**
	 @WHAT
	   Return list of values of the Range
	   
	 @PARAMETERS
	   @param:values vector of integers that will store
	   the values
	     
	 */
	void values(std::vector<int>& values);
	
};

} // end of namespace essential

} // end of namespace ez

#endif /* ESSENTIAL_RANGE_H_ */
