#include "essential/import.h"
#include "extensions/import.h"
using namespace eze;

void describe(text name, Range& r) {
	cout << Terminal::line_type_3 << endl;
	cout << name << "=" <<  r << endl;
	cout << name << ".size()=" << r.size() << endl;
	 
	cout << name << " values=";
	for (auto x : r) {
		cout << x << " ";
	}
	cout << endl;
	cout << Terminal::line_type_3 << endl;
} 

int main() {


	Range r1(-10,10);
	describe("r1", r1);

	Range r2(0,10);
	describe("r2", r2);

	Range r3(8,20);
	describe("r3", r3);
	
	Range r4(40,50,2);
	describe("r4", r4);

/*	
	cout << Terminal::line_type_3 << endl;
	cout << "is_inside" << endl;
	cout << Terminal::line_type_3 << endl;
	cout << "r1.is_inside(r2)=" << r1.is_inside(r2) << endl;
	cout << "r1.is_inside(r3)=" << r1.is_inside(r3) << endl;
	cout << "r1.is_inside(r4)=" << r1.is_inside(r4) << endl;
*/

	cout << Terminal::line_type_3 << endl;
	cout << "to_range" << endl;
	cout << Terminal::line_type_3 << endl;
	for (integer i = 0; i < static_cast<integer>(r3.size()); ++i) {
		cout << "index=" << i << " -> " << r3.to_range(i) << " of r3" << endl;
	}
	
	cout << Terminal::line_type_3 << endl;
	cout << "to_index of r3" << endl;
	cout << Terminal::line_type_3 << endl;
	for (auto x : r3) {
		cout << "value=" << x << " -> " << r3.to_index(x) << endl;
	}
	
	std::vector<integer> values;
	r1.values(values);

	cout << "r1 (get_values) = ";
	ezx::print(	cout, values );
	cout << endl;


	
	return EXIT_SUCCESS;
}
