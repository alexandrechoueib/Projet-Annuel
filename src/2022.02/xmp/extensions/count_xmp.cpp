#include "extensions/import.h"
#include "objects/import.h"
using namespace ezo;
using namespace ezx;

bool is_even( integer v ) {
	return (v & 1) == 0;
}

class IotaGenerator {
public:
	integer _value;
	
	IotaGenerator( integer v ) : _value( v ) {
	}
	
	integer operator()() {
		return _value++;
	}
	
};

int main() {

	ezo::Array<integer> a(1,10);
	IotaGenerator g(10);
	
	ezx::generate<Array<integer>,integer,IotaGenerator>(a, g);
	
	cout << "a=" << a << endl;
	
	natural nbr = ezx::count_if( a, is_even );
	cout << "nbr = " << nbr << endl;
	
	integer sum1 = ezx::sum( a, 0 );
	cout << "sum1 = " << sum1 << endl;
	
	bool has10 = ezx::contains( a, 10 );
	cout << "has 10 = " << has10 << endl;
	bool has9 = ezx::contains( a, 9 );
	cout << "has 9 = " << has9 << endl;
	
	bool has10b = ezx::contains( a, 10, X_BINARY_SEARCH );
	cout << "has 10 = " << has10b << endl;
	bool has9b = ezx::contains( a, 9, X_BINARY_SEARCH );
	cout << "has 9 = " << has9b << endl;
	
	return EXIT_SUCCESS;
}

