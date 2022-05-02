#include "essential/import.h"
#include "io/import.h"
#include "extensions/import.h"
using namespace eze;
using namespace ezio;
using namespace ezx;

int main() {  

	CSVFile csv("data/csv_example.csv");
	csv.read(";");
 
	std::vector<std::vector<text>> data = csv.data();
	  
	cout << "number of lines=" << csv.nbr_lines() << endl;
	
	cout << eze::Terminal::line_type_3 << endl;
	cout << "browse with integer for loop" << endl;
	cout << eze::Terminal::line_type_3 << endl;
	
	for (u32 i = 0; i < data.size(); ++i) {
		for (u32 j = 0; j < data[i].size(); ++j) {
			cout << data[i][j] << " * "; 
		}
		cout << endl;
	}
	
	cout << eze::Terminal::line_type_3 << endl;
	cout << "browse with CSVFile::iterator" << endl;
	cout << eze::Terminal::line_type_3 << endl;
	
	CSVFile::iterator iter = csv.begin();
	while (iter != csv.end()) {
		ezx::print(cout, *iter);
		++iter;
	}

	return EXIT_SUCCESS;
}

