/*
 * out.h
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

#ifndef EXTENSIONS_OUT_H_
#define EXTENSIONS_OUT_H_

#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <map>


namespace ez {

namespace extensions {

#define debug_container(v) \
		std::cerr << #v << "(" << v.size() << ")="; \
		stda::print(std::cerr, v);

#define debug_var(v) \
		std::cerr << #v << "=" << v << std::endl;

#define debug_array(v,sz) \
		std::cerr << #v << "=[ "; \
		for (size_t i = 0; i<sz; ++i) std::cerr << v[i] << " "; \
		std::cerr << "]" << std::endl;


#define echo(x) \
		std::cout << #x << "=" << x << std::endl;

#define dump(x) \
		std::cerr << #x << "=" << x << std::endl;


/**
 * print contents of container
 * @param out output stream
 * @param c STL container
 * @param delim delimiter
 */
template<typename Container>
void print( std::ostream& out, Container& c, std::string delim=" " ) {

	if (c.size() > 0) {
		auto it = c.begin();
		out << "[";
		out << *it++;
		while (it != c.end()) {
			out << delim << *it;
			++it;
		}
		out << "]" << std::endl;
	}
	
}


/**
 * print contents of container with formatted data
 * @param out output stream
 * @param c STL container
 * @param f Formatter, a class that redefines the () operator and
 * takes as a parameter the output stream and the data to format
 * @param delim delimiter
 */
template<typename Container, typename Formatter>
void print_fmt( std::ostream& out, Container& c, Formatter f, std::string delim=" " ) {

	if (c.size() > 0) {
		auto it = c.begin();
		out << "[";
		f(out, *it++);
		while (it != c.end()) {
			out << delim ;
			f(out, *it);
			++it;
		}
		out << "]" << std::endl;
	}
	
}

/**
 * print contents of container of type map
 * @param out output stream
 * @param m STL map container
 * @param delim delimiter
 */
template<typename K, typename Container>
void print( std::ostream& out, std::map<K,Container>& m, std::string delim=" " ) {

	for (auto e : m) {
		out << e.first << ": ";
		ez::extensions::print<Container>( out, e.second, delim );
	}

}


/**
 * print contents of container with formatted data
 * @param out output stream
 * @param m STL container of type map
 * @param f Formatter, a class that redefines the () operator and
 * takes as a parameter the output stream and the data to format
 * @param delim delimiter
 */
template<typename K, typename Container, typename Formatter>
void print_fmt( std::ostream& out, std::map<K,Container>& m, Formatter f, std::string delim=" " ) {

	for (auto e : m) {
		out << e.first << ": ";
		print_fmt( out, e.second, f, delim );
	}

}

} // end of namespace extensions

} // end of namespace ez


#endif /* EXTENSIONS_OUT_H_ */
