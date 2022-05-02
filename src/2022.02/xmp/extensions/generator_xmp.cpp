#include "extensions/import.h"
using namespace ezx;

int main() {

	cout << "Generator<int> gen(1,1)" << endl;
	Generator<int> gen1(1,1);
	
	for (int i = 1; i <= 10; ++i) {
		cout << gen1() << endl;
	}

	cout << "Generator<int> gen(1,2)" << endl;
	Generator<int> gen2(1,2);
	
	for (int i = 1; i <= 10; ++i) {
		cout << gen2() << endl;
	}

	cout << "Generator<real> gen(0.0,3.14)" << endl;
	Generator<real> gen3(0.0,3.14);
	
	for (int i = 1; i <= 10; ++i) {
		cout << gen3() << endl;
	}

	return EXIT_SUCCESS;
}

