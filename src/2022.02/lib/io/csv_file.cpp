/*
 * csv_file.cpp
 *
 *  Created on: Mar 30, 2018
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "io/csv_file.h"

using namespace ez::io;


CSVFile::CSVFile( text file_name ) {

	_file_name = file_name;
	
}


void CSVFile::read( text field_delimiter, text string_delimiter ) {

	std::ifstream ifs( _file_name );
	std::string line;

	if (ifs.bad()) {
		notify( "could not open CSV file" );
	}
	
	while (ifs.good()) {
	
		getline( ifs, line );
		ez::essential::TextUtils::trim( line );
		if (line.size() > 0) {
			std::vector<std::string> words;
			ez::essential::TextUtils::explode( line, words, field_delimiter );
			for (auto& s : words) {
				ez::essential::TextUtils::trim(s);
				if (ez::essential::TextUtils::starts_with( s, string_delimiter, true )) {
					ez::essential::TextUtils::ends_with( s, string_delimiter.c_str(), true );
				}
			}
			_data.push_back(words);
			
		}

	}

}



