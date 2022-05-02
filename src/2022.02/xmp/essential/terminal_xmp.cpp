#include "essential/import.h"

using namespace eze;



int main() {
	cout << Terminal::line_type_1 << endl;
	cout << Terminal::b_bold << "Terminal" << Terminal::e_bold << endl;
	cout << Terminal::line_type_1 << endl;
	cout << Terminal::b_underline << "First line" << Terminal::e_underline << endl;
	cout << Terminal::underline("Second line") << endl;
	cout << Terminal::bold("end") << endl;
	
	cout << Terminal::line_type_1 << endl;
	cout << "CHAPTER" << endl;
	cout << Terminal::line_type_1 << endl;

	cout << Terminal::line_type_2 << endl;
	cout << "Section" << endl;
	cout << Terminal::line_type_2 << endl;
	
	cout << Terminal::line_type_3 << endl;
	cout << "Subsection" << endl;
	cout << Terminal::line_type_3 << endl;
	
	cout << Terminal::line_type_4 << endl;
	cout << "Subsubsection" << endl;
	cout << Terminal::line_type_4 << endl;
	
	cout << Terminal::line_type_5 << endl;
	cout << "Paragraph" << endl;
	cout << Terminal::line_type_5 << endl;


	
	return EXIT_SUCCESS;	
}
