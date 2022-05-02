#include "maths/import.h"
using namespace ezm;

int main() {
	std::vector<int> notes1 = { 7, 8, 2, 10, 20, 15, 13, 11, 10, 10, 6, 11 };
	Series<int> series1( notes1 );
	Statistics<int, real> stats1( series1 );
	stats1.compute();
	cout << "===== series1 = " << series1 << endl;
	cout << stats1 << endl;

	std::vector<int> notes2 = { 9, 9, 9, 9, 11, 11, 11, 11 };
	Series<int> series2( notes2 );
	Statistics<int, real> stats2( series2 );
	stats2.compute();
	cout << "===== series2 = " << series2 << endl;
	cout << stats2 << endl;

	std::vector<int> notes3 = { 10, 10, 10, 10 };
	Series<int> series3( notes3 );
	Statistics<int, real> stats3( series3 );
	stats3.compute();
	cout << "===== series3 = " << series3 << endl;
	cout << stats3 << endl;
	
	return EXIT_SUCCESS;
}

