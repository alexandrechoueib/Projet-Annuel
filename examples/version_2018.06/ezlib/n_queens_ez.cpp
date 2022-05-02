/*
 * n_queens_ez.cpp
 *
 *  Created on: Aug 2, 2017
 *      Author: Jean-Michel Richer
 *
 * This program computes the solutions of the N-Queens problems.
 * Use of ez::arguments, ez::logging and Array of ez::objects
 */


#include "arguments/import.h"
#include "essential/import.h"
#include "logging/import.h"
#include "objects/import.h"

// possible command line parameters
integer dimension = 8;
boolean quiet = false;
long_natural nbr_solutions = 0;
eze::CPUTimer timer;

ezo::Array<integer> columns;
ezo::Array<integer> diagonals1;
ezo::Array<integer> diagonals2;

void print_solution(ostream& out) {
	for (int r = 1; r <= dimension; ++r) {
		int pos = columns[r];
		for (int c = 1; c <= dimension; ++c) {
			if (c == pos) {
				cout << "Q";
			} else {
				cout << ".";
			}
		}
		cout << endl;
	}
}

/**
 * Check whether a queen can be put in given row.
 * For this we verify that the queen won't threaten
 * a queen that was positioned previously.
 * @return true if it is possible, false otherwise
 */
bool can_place(integer row, integer col) {
	integer value = columns[col] | diagonals1[row + col] | diagonals2[-row+col];
	return (value == 0);
}

void place(integer queen) {
	// if we have reach number of queens to place
	// then we have a solution
	if (queen > dimension) {
		++nbr_solutions;
		if (!quiet) {
			cout << "solution: " << nbr_solutions << endl;
			print_solution(cout);
		}
		return ;
	}

	// otherwise try to place queen in columns 1 to dimension
	for (int col = 1; col <= dimension; ++col) {
		// if we can place queen in a column then
		// continue with next queen
		if (can_place(queen, col)) {
			if (!quiet) {
				cout << "put queen " << queen << " in column " << endl;
			}

			// add chess board information about queen position
			columns[col] = queen;
			diagonals1[queen + col] = queen;
			diagonals2[-queen + col] = queen;

			place(queen+1);

			// remove information about queen position
			columns[col] = 0;
			diagonals1[queen + col] = 0;
			diagonals2[-queen + col] = 0;

		}
	}
}





/**
 * program
 * we can specify the dimension of the board using command line argument
 * as "--dimension=5" or "-d 5" or "--quiet" or "-q" to have a non verbose
 * output
 */
int main(int argc, char *argv[]) {
	eza::ArgumentParser parser(argv[0],
			"find all solutions to the N-Queens problem",
			argc, argv);

	try {
		parser.add_integer("dimension", 'd', &dimension, "size of board");
		parser.add_flag("quiet", 'q', &quiet, "quiet mode");

		parser.parse();
	} catch(Exception& e) {
		parser.report_error(e);
	}

	cout << "dimension=" << dimension << endl;
	columns.resize(eze::Range(1, dimension));
	diagonals1.resize(eze::Range(1, 2*dimension));
	diagonals2.resize(eze::Range(-dimension+1, dimension-1));
	columns.fill(0);
	diagonals1.fill(0);
	diagonals2.fill(0);

	timer.start();
	place(1);
	timer.stop();

	cout << "it took " << timer << " to find all solutions" << endl;
	cout << "found " << nbr_solutions << " solution(s)" << endl;

	return EXIT_SUCCESS;
}






