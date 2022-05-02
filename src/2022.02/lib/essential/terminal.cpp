/*
 * terminal.cpp
 *
 *  Created on: May 25, 2015
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "essential/terminal.h"

using namespace ez::essential;

string Terminal::line_type_1 = "############################################################";
string Terminal::line_type_2 = "************************************************************";
string Terminal::line_type_3 = "============================================================";
string Terminal::line_type_4 = "------------------------------------------------------------";
string Terminal::line_type_5 = "____________________________________________________________";

string Terminal::b_underline = "\33[4m";
string Terminal::e_underline = "\033[0m";
string Terminal::b_bold = "\033[1m";
string Terminal::e_bold = "\033[0m";


void Terminal::press_return() {

	cout << endl << "Press [return]... ";
	cin.get();
	cout << endl;
	
}


text Terminal::bold( text s ) {

	string r = b_bold + s + e_bold;
	return r;
	
}


text Terminal::underline( text s ) {

	string r = b_underline + s + e_underline;
	return r;
	
}
