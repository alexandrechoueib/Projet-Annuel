/*
 * sum.h
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

#ifndef EXTENSIONS_SUM_H_
#define EXTENSIONS_SUM_H_

#include <numeric>

namespace ez {

namespace extensions {

	/**
	 * definition of the sum of all elements in a container
	 * (ex: vector<int>) based on accumulate.
	 */
	template<typename Container, typename DataType>
	DataType sum( Container& c, DataType zero ) {
	
		return std::accumulate( c.begin(), c.end(), zero );
	
	}


	/**
	 * definition of the sum of elements in a container
	 * with extraction of property
	 * @param c a container for which the operator += exists
	 * @return sum with type of getter
	 */
	template<typename Container, typename Getter>
	typename Getter::value_type sum( Container& c, Getter& g, typename Getter::value_type zero ) {
	
		typename Getter::value_type s = zero;
		auto it = c.begin();
		while (it != c.end()) {
			s += g.get( *it );
			++it;
		}
		
		return s;
		
	}


	/**
	 * definition of the sum of all elements in a container
	 * (ex: map<K,Container>) based on accumulate.
	 */
	template<typename K, typename Container>
	typename Container::value_type sum( std::map<K,Container>& m, K key ) {
	
		return std::accumulate( m[ key ].begin(), m[ key ].end(), 0 );
	}


	/**
	 * definition of the sum of all elements in a container
	 * (ex: map<K,Container>) based on accumulate.
	 */
	template<typename K, typename Container, typename Getter>
	typename Getter::value_type sum( std::map<K,Container>& m, K key, Getter& g ) {
	
		typename Getter::value_type s = 0;
		auto it = m[ key ].begin();
		while (it != m[ key ].end()) {
			s += g.get( *it );
			++it;
		}
		
		return s;
		
	}


}

}

#endif /* EXTENSIONS_SUM_H_ */
