#include "essential/import.h"

using namespace eze;

text separator = "======================================";

int main() {
	text s = "hello wordl!";
	
	cout << "Format::left(s,40)   =[" << Format::left(s, 40) << "]" <<  endl;
	cout << "Format::center(s,40) =[" << Format::center(s, 40) << "]" <<  endl;
	cout << "Format::right(s,40)  =[" << Format::right(s, 40) << "]" <<  endl;

	cout << separator << endl;
	
	cout << "Format::bin(249)     =[" << Format::bin(249) << "]" << endl;	
	cout << "Format::oct(249)     =[" << Format::oct(249) << "]" << endl;	
	cout << "Format::hex(249)     =[" << Format::hex(249) << "]" << endl;	
	
	cout << separator << endl;
	
	cout << "Format::fp(3.1415926, 10, 5)  =[" << Format::fp(3.1415926, 10, 5) << "]" << endl;
	cout << "Format::fp(3.1415926, 0, 7)   =[" << Format::fp(3.1415926, 0, 7) << "]" << endl;
	
	return EXIT_SUCCESS;	
}
