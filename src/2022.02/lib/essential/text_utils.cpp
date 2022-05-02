/*
 * text_utils.cpp
 *
 *  Created on: Apr 2, 2015
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "essential/text_utils.h"

using namespace ez::essential;


void TextUtils::inspect( std::ostream& out, text& s ) {

	ios::fmtflags old_settings = out.flags();
	for (text::size_type pos = 0; pos != s.size(); ++pos) {
		integer value = static_cast<int>( s[ pos ] );
		out << setw( 3 ) << pos << ": " << setw( 3 ) << value << " ";
		if (value > 32) {
			out << s[ pos ];
		}
		out << endl;
	}
	
	out.flags( old_settings );
	
}


void TextUtils::trim_right( text& s, const text& pattern ) {

	text::size_type pos = s.find_last_not_of( pattern );
	if (pos != text::npos)
		s.erase( pos + 1, text::npos );

}


void TextUtils::trim_left( text& s, const text& pattern ) {

	text::size_type pos = s.find_first_not_of(pattern);
	if (pos != text::npos) {
		s.erase( 0, pos );
	} else {
		if (s.size()>0) {
			s.clear();
		}
	}

}


void TextUtils::trim( text& s, const text& pattern ) {

	TextUtils::trim_right( s );
	TextUtils::trim_left( s );

}

void TextUtils::erase( text& s, const text& pattern ) {

	text::size_type last_pos = 0;
	text::size_type pos = s.find_first_of( pattern );
	
	while (pos!=text::npos) {
		last_pos=s.find_first_not_of( pattern, pos );
		s.erase( pos, last_pos-pos );
		last_pos = pos;
		pos = s.find_first_of( pattern, last_pos );
	}

}


void TextUtils::remove_spaces( text& s ) {

	TextUtils::erase( s );
	
}


void TextUtils::lower( text& s ) {

	std::transform( s.begin(), s.end(), s.begin(), (int(*)(int)) std::tolower );

}


void TextUtils::upper( text& s ) {

	std::transform( s.begin(), s.end(), s.begin(), (int(*)(int)) std::toupper );

}


void TextUtils::lower( char *s ) {

	while (*s) {
		*s = (char) tolower( *s );
		++s;
	}

}


void TextUtils::upper( char *s ) {

	while (*s) {
		*s = (char) toupper( *s );
		++s;
	}

}


void TextUtils::explode( text& s, std::vector<text>& v, const text& delim ) {

	// Skip delimiters at beginning.
	text::size_type lastPos = s.find_first_not_of( delim, 0 );
	// Find first "non-delimiter".
	text::size_type pos = s.find_first_of( delim, lastPos );

	while (text::npos != pos || text::npos != lastPos) {
		// Found a token, add it to the vector.
		v.push_back( s.substr( lastPos, pos - lastPos ) );
		// Skip delimiters.  Note the "not_of"
		lastPos = s.find_first_not_of( delim, pos );
		// Find next "non-delimiter"
		pos = s.find_first_of( delim, lastPos );
	}
	
}


void TextUtils::implode( text& s,  std::vector<text>& v, const text& delim ) {

	natural i;

	for (i = 0; i < v.size(); ++i) {
		s += v[ i ];
		if (i != (v.size()-1)) {
			if (delim.size() > 0) s += delim;
		}
	}
	
}


void TextUtils::implode( text& s, std::vector<char>& v, const text& delim ) {

	natural i;

	for (i = 0; i < v.size(); ++i) {
		s+=v[i];
		if (i!=(v.size()-1)) {
			if (delim.size() > 0) s += delim;
		}
	}
	
}


bool TextUtils::starts_with( char *s, char *st_str, bool remove_flag ) {

	natural length = strlen( st_str );
	if (strncmp(s, st_str, length) == 0) {
		if (remove_flag == true) {
			memmove( s, &s[ length ], strlen( s ) - length + 1 );
		}
		return true;
	}
	return false;

}


bool TextUtils::starts_with( text& s, text& st_str, bool remove_flag ) {

	bool result = (strncmp( s.c_str(), st_str.c_str(), st_str.size() )==0) ? true : false;
	if (result & remove_flag) {
		s.erase( 0, st_str.size() );
	}
	return result;

}


bool TextUtils::starts_with( text& s, char *st_str ) {

	return (strncmp( s.c_str(), st_str, strlen( st_str ) )==0) ? true : false;

}


bool TextUtils::starts_with( text& s, const char *st_str ) {

	return (strncmp( s.c_str(), const_cast<char *>( st_str ), strlen( st_str ) )==0) ? true : false;

}


bool TextUtils::ends_with( text& s, text& st_str, bool remove_flag ) {

	ensure( st_str.size() > 0 );
	if (s.size() < st_str.size()) return false;
	natural length = st_str.size();
	bool result = (strcmp( &s.c_str()[ s.size()-length ], st_str.c_str() )==0) ? true : false;
	if (result & remove_flag) {
		s.resize( s.length() - st_str.size() );
	}
	
	return result;
	
}


bool TextUtils::ends_with( text& s, char *st_str, bool remove_flag ) {

	natural length = static_cast<natural>( strlen(st_str) );
	ensure( length > 0 );
	
	if (s.size() < length) return false;
	
	bool result = (strcmp( &s.c_str()[ s.size()-length ], st_str)==0) ? true : false;
	if (result & remove_flag) {
		s.resize( s.length() - length );
	}
	return result;
	
}


bool TextUtils::ends_with(text& s, const char *st_str, bool remove_flag) {

	natural length = static_cast<natural>( strlen(st_str) );
	ensure(length > 0);
	if (s.size() < length) return false;
	bool result = (strcmp( &s.c_str()[ s.size()-length ], const_cast<char *>( st_str ))==0) ? true : false;
	if (result & remove_flag) {
		s.resize( s.length() - length );
	}
	return result;
	
}


void TextUtils::replace_char( text& s, char src, char repl ) {

	text::size_type i;
	for (i = 0; i != s.length(); ++i) {
		if (s[ i ] == src) {
			s[ i ] = repl;
		}
	}
	
}


size_t TextUtils::position_of( text& s, text& st_str ) {

	text::size_type pos = s.find( st_str );
	if (pos == text::npos) {
		return 0;
	} else {
		return static_cast<size_t>( pos + 1 );
	}

}


bool TextUtils::contains( text& s, text& st_str ) {

	text::size_type pos = s.find( st_str );
	if (pos == text::npos) {
		return false;
	} else {
		return true;
	}
	
}


size_t TextUtils::position_of( text& s, char *st_str ) {

	text::size_type pos = s.find( st_str );
	if (pos == text::npos) {
		return 0;
	} else {
		return static_cast<size_t>( pos + 1 );
	}
	
}


void TextUtils::remove_at( text& s, natural pos, natural length ) {

	ensure(pos > 0);
	ensure(length > 0);
	s.erase(pos - 1, length);
	
}


void TextUtils::replace( text& s, string pattern, string replace ) {

	for(size_t pos = 0; ; pos += replace.length() ) {
		// Locate the substring to replace
		pos = s.find( pattern, pos );
		if( pos == string::npos ) break;
		// Replace by erasing and inserting
		s.erase( pos, pattern.length() );
		s.insert( pos, replace );
	}

}
