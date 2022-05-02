/*
 * algorithms.h
 *
 *  Created on: Apr 8, 2017
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

#ifndef EXTENSIONS_ALGORITHMS_H_
#define EXTENSIONS_ALGORITHMS_H_

#include "essential/import.h"
using namespace eze;

namespace ez {

namespace extensions {
/**
 * Extensions of algorithms to containers. Instead of using
 * the .begin(), .end() methods we use the name of the container.
 * Example:
 *   std::vector<int> v;
 *
 *  with STL: std::count(v.begin(), v.end(), x);
 *  with EZLib: ezx::count(v, x) or ez::extensions::count(v, x)
 */

/**
 * count extension for container
 */
template<typename Container, typename DataType>
natural count( Container& c, DataType value ) {

	return std::count( c.begin(), c.end(), value );
	
}


/**
 * count if extension for container
 */
template<typename Container, typename UnaryFunction>
natural count_if( Container& c, UnaryFunction f ) {

	return std::count_if( c.begin(), c.end(), f );

}


/**
 * apply non modifying function on container
 */
template<typename Container, typename UnaryFunction>
UnaryFunction for_each( Container& c, UnaryFunction f ) {

	return std::for_each( c.begin(), c.end(), f );

}


/**
 * apply modifying function on container
 */
template<typename Container, class UnaryOperation>
void transform( Container &c, UnaryOperation op ) {

	std::transform( c.begin(), c.end(), c.begin(), op );

}


/**
 * fill container with given value, the container must
 * already have some size
 */
template<typename Container, class DataType>
void fill( Container &c, DataType value ) {

	std::fill( c.begin(), c.end(), value );

}


/**
 * fill container with given value, the container must
 * already have some size
 */
template<typename Container, class DataType, class Generator>
void generate( Container &c, Generator& g ) {

	std::generate( c.begin(), c.end(), g );

}


/**
 * find extension
 */
template<typename Container>
typename Container::iterator find( Container& c, typename Container::value_type v ) {

	auto it = std::find( c.begin(), c.end(), v );
	return (it != c.end());

}


/**
 * check if all elements of container are different
 * @param c Container that is checked
 * @return true if all elements are different, false otherwise
 */
template<typename Container>
bool all_diff( Container& c ) {

	auto end = c.end() - 1;
	for (auto i = c.begin(); i < end; ++i) {
		for (auto j = i+1; j < c.end(); ++j) {
			if ((*i) == (*j)) return false;
		}
	}
	
	return true;

}


/**
 * check if all elements of container are different
 * @param c Container that is checked
 * @param f binary function to compare two elements
 * @return true if all elements are different, false otherwise
 */
template<typename Container, typename BinaryFunction>
bool all_diff( Container& c, BinaryFunction f ) {

	auto end = c.end()-1;
	for (auto i = c.begin(); i < end; ++i) {
		for (auto j = i + 1; j < c.end(); ++j) {
			if (f( *i, *j )) return false;
		}
	}
	
	return true;
	
}


/**
 * check if all elements of container are different and return iterator
 * on first element that appears twice
 * @param c Container that is checked
 * @return iterator on first element that appears twice or c.end() if
 * all elements are different
 */
template<typename Container>
typename Container::iterator all_diff_pos( Container& c ) {

	auto end = c.end()-1;
	for (auto i = c.begin(); i < end; ++i) {
		for (auto j = i + 1; j < c.end(); ++j) {
			if ((*i) == (*j)) return i;
		}
	}

	return c.end();
	
}


/**
 * check if all elements of container have an equivalent value
 * @param c Container that is checked
 * @return iterator on first element that appears twice or c.end() if
 * all elements are different
 */
template<typename Container>
bool all_match( Container& c1, Container& c2 ) {

	if (c1.size() * c2.size() == 0) return false;
	if (c1.size() != c2.size()) return false;
	
	return std::is_permutation( c1.begin(), c1.end(), c2.begin(), c2.end() );
	
}


template<typename Container, class BinaryPredicate>
bool all_match( Container& c1, Container& c2, BinaryPredicate pred ) {

	if (c1.size() * c2.size() == 0) return false;
	if (c1.size() != c2.size()) return false;

	return std::is_permutation( c1.begin(), c1.end(), c2.begin(), c2.end(), pred );

}


/**
 * remove all elements from container
 * @param c container from which to remove elements
 * @param elements container that has elements to remove
 */
template<typename Container>
void remove_all_from( Container& c, Container& elements_to_remove ) {

	for (auto x : elements_to_remove) {
		auto it = std::find( c.begin(), c.end(), x );
		if (it != c.end()) {
			c.erase( it );
		}
	}
	
}


template<typename Container, typename Predicate>
void remove_if(Container& c, Predicate p) {

}


/**
 * redefinition of sort for a container
 */
template<typename Container>
void sort(Container& c) {

	std::sort( c.begin(), c.end() );
	
}


/**
 * redefinition of sort for a container
 */
template<typename Container, typename Comparator>
void sort_comparator( Container& c, Comparator comp) {

	std::sort( c.begin(), c.end(), comp );
	
}


/**
 * definition of reverse sort for a container
 */
template<typename Container>
void rsort(Container& c) {

	std::sort( c.rbegin(), c.rend() );
	
}


/**
 * definition of reverse sort for a container
 */
template<typename Container, typename Comparator>
void rsort_comparator( Container& c, Comparator comp) {

	std::sort( c.rbegin(), c.rend(), comp );

}

/**
 * redefinition of sort with getter 
 */
template<typename Container, typename Getter>
void sort( Container& c, Getter& g ) {

	typedef typename Container::value_type T;
	
	std::sort( c.begin(), c.end(), [&g]( const T& a, const T&b ){
		return g.get( a ) < g.get( b );
	});
	
}


/**
 * redefinition of reverse sort with getter
 */
template<typename Container, typename Getter>
void rsort( Container& c, Getter& g ) {

	typedef typename Container::value_type T;
	
	sort( c.rbegin(), c.rend(), [&g](const T& a, const T&b ){
		return g.get( a ) < g.get( b );
	});
	
}

/**
 * Compare two container and they must have exact values
 * at the same index in the container
 */
template<typename Container>
bool are_equal( Container& c1, Container& c2 ) {

	auto i1 = c1.begin();
	auto i2 = c2.begin();
	while ((i1 != c1.end()) && (i2 != c2.end())) {
		if (*i1 != *i2) return false;
		++i1;
		++i2;
	}

	return (i1 = c1.end()) && (i2 = c2.end());

}


template<typename Container>
bool are_equal(Container& c1, Container& c2, natural size) {
	auto i1 = c1.begin();
	auto i2 = c2.begin();
	while ((i1 != c1.end()) && (i2 != c2.end()) && (size != 0)) {
		if (*i1 != *i2) return false;
		++i1;
		++i2;
		--size;
	}
	return (size == 0);
}


} // end of namespace extensions

} // end of namespace ez


#endif /* EXTENSIONS_ALGORITHMS_H_ */
