/*
 * csv_file.h
 *
 *  Created on: Mar 30, 2018
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

#ifndef IO_CSV_FILE_H_
#define IO_CSV_FILE_H_

#include <fstream>

#include "essential/exception.h"
#include "essential/text_utils.h"
#include "essential/types.h"
#include "maths/matrix.h"

using namespace ez::essential;

namespace ez {

namespace io {

class CSVFile {
protected:
	std::string _file_name;
	std::vector<std::vector<std::string>> _data;

public:
	typedef std::vector<std::vector<std::string>>::iterator iterator;
	
	/**
	 * Constructor
	 * @param file_name (text) name of file
	 */
	CSVFile( text file_name );


	/**
	 * Read csv file
	 */
	void read( text field_delimiter=",", text string_delimiter="\"" );
	
	/**
	 * Return number of lines
	 */
	natural nbr_lines() {
	
		return _data.size();
		
	}
	
	/**
	 * Return access to data
	 */
	std::vector<std::vector<text>>& data() {
	
		return _data;
		
	}
	
	/**
	 * Return iterator on first row of data
	 */
	iterator begin() {
	
		return _data.begin();
		
	}
	
	
	/**
	 * Return iterator on last row of data
	 */
	iterator end() {
	
		return _data.end();
		
	}
	

};

} // end of namesapce io

} // end of namespace ez

#endif /* IO_CSV_FILE_H_ */
