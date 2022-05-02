#include "essential/import.h"
#include "objects/import.h"
using namespace ezo;

int main() {
	
	cout << "min=" << Boolean::min() << endl;
	cout << "max=" << Boolean::max() << endl;
	
	Boolean a(true), b(true), c(false), d(false);
	Boolean e1, e2, e3, e4;
	
	cout << "a=" << a << endl;
	cout << "b=" << b << endl;
	cout << "c=" << c << endl;
	cout << "d=" << d << endl;
	
	e1 = a + b;
	cout << "e1 = a + b = " << e1 << endl;
	e2 = a + c;
	cout << "e2 = a + c = " << e2 << endl;
	e3 = c + d;
	cout << "e3 = c + d = " << e3 << endl;
	
	e4 = a + (c * d);
	cout << "e4 = a + c * d = " << e4 << endl;
	e4 = a | (c & d);
	cout << "e4 = a | c & d = " << e4 << endl;
	e4 = -a | (c & d);
	cout << "e4 = -a | c & d = " << e4 << endl;
	e4 = ~a | (c & d);
	cout << "e4 = ~a | c & d = " << e4 << endl;
	
	cout << "_bool_('a')=" << Boolean::_bool_('a') << endl;
	cout << "_bool_('\\0')=" << Boolean::_bool_('\0') << endl;
	
	cout << "_bool_(0)=" << Boolean::_bool_(0) << endl;
	cout << "_bool_(1)=" << Boolean::_bool_(1) << endl;
	cout << "_bool_(255)=" << Boolean::_bool_(255) << endl;
	
	cout << "_bool_(0.0)=" << Boolean::_bool_(0.0) << endl;
	cout << "_bool_(3.14)=" << Boolean::_bool_(3.14) << endl;
	
	cout << "_bool_(\"true\")=" << Boolean::_bool_("true") << endl;
	cout << "_bool_(\"false\")=" << Boolean::_bool_("false") << endl;
	
	

	return EXIT_SUCCESS;
}
