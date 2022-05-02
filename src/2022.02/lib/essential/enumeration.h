/*
 * enumeration.h
 *
 *   Created on: 30 août 2020
 *  Modified on: Feb, 2022
 *       Author: Jean-Michel Richer
 */

#ifndef ESSENTIAL_ENUMERATION_H_
#define ESSENTIAL_ENUMERATION_H_

#include "essential/ensure.h"
#include "essential/exception.h"
#include "essential/language.h"
#include "essential/scalar_types.h"

namespace ez {

namespace essential {

/**
 @CLASS
   Class used to define an enumeration with
   integer values and related labels (string representation).
   
 */
class Enumeration {
public:
	typedef Enumeration self;

	// values that compose the enumeration
	std::vector<integer> _values;
	
	// string representation of values
	std::vector<std::string> _labels;
	
	// number of values
	natural _size;


public:
	/**
	 @WHAT
	   Default constructor with no values
	    
	 */
	Enumeration() {
	
		_size = 0;
		
	}

	/**
	 @WHAT
	   Constructor given values and labels.
	   
	 */
	Enumeration( std::vector<integer> &values, 
		std::vector<std::string> &labels ) {
		
		ensure( values.size() == labels.size() );
		_size = static_cast<integer>( values.size() );
		
		copy( values.begin(), values.end(), back_inserter( _values ) );
		copy( labels.begin(), labels.end(), back_inserter( _labels ) );
		
	}

	/**
	 @WHAT
	   Copy constructor
	   
	 */
	Enumeration( const Enumeration& r ) {
	
		_size = r._size;
		copy( r._values.begin(), r._values.end(), back_inserter( _values ) );
		copy( r._labels.begin(), r._labels.end(), back_inserter( _labels ) );
	}


	/**
	 @WHAT
	   Overloading of assignment operator
	   
	 */
	self& operator=( const self& r ) {
	
		if (&r != this) {
			_size = r._size;
			_values.clear();
			_labels.clear();
			copy( r._values.begin(), r._values.end(), back_inserter( _values ) );
			copy( r._labels.begin(), r._labels.end(), back_inserter( _labels ) );
		}
		
		return *this;
		
	}

	/**
	 @WHAT
	   Return number of elements in the Enumeration
	   
	 */
	const natural size() {
	
		return _size;
		
	}


	/**
	 @WHAT
	   Check whether value @param:x is part of the Enumeration.
	
	 @RETURN
	   true if @param:x is a value of the Enumeration, false
	   otherwise
	   
	 */
	bool contains( integer x ) {
	
		auto it = find( _values.begin(), _values.end(), x );
		return (it != _values.end());
		
	}

	/**
	 @WHAT
	   Return value that corresponds to label
	   
	 */
	integer to_value( text s ) {
	
		for (integer i = 0; i < static_cast<integer>( _labels.size() ); ++i) {
			if (_labels[ i ] == s) {
				return _value[ i ];
			}
		}
		
		notify("string identifier " << s << " is not in Enumeration");
		return 0;
		
	}

	/**
	 @WHAT
	   Return label that corresponds to integer value in the
	   Enumeration.
	   
	 */
	text to_label( integer value ) {
	
		for (integer i = 0; i < static_cast<integer>(_values.size()); ++i) {
			if (_values[ i ] == value) return _labels[ i ];
		}
		
		notify("value " << value << " is not in Enumeration");
		return "";
		
	}

	typedef std::vector<integer>::iterator iterator;
	
	/**
	 @WHAT
	   Definition of begin iterator
	   
	 */
	iterator begin() {
	
		return _values.begin();
		
	}

	/**
	 @WHAT
	   Definition of end iterator
	   
	 */
	iterator end() {
	
		return _values.end();
		
	}

	/**
	 @WHAT 
	   Return values of the Enumeration
	   
	 */
	std::vector<integer>& values() {
	
		return _values;
		
	}

	/**
	 @WHAT
	   Print contents of Enumeration

	 */
	std::ostream& print( std::ostream& out ) {
	
		for (natural i = 0; i < _values.size(); ++i) {
			out << "(" << _values[i] << ",";
			out << "\"" << _labels[i] << "\")";
		}
		
		return out;
		
	}

	/**
	 @WHAT
	   Overloading of output operator
	   
	 */
	friend std::ostream& operator<<(std::ostream& out, Enumeration& obj) {

		return obj.print(out);

	}

};

} // end of namespace essential

} // end of namespace ez



#endif /* SRC_VERSION_2019_08_ESSENTIAL_ENUMERATION_H_ */
