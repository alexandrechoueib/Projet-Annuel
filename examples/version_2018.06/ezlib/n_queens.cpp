/*
 * n_queens.cpp
 *
 *  Created on: Aug 1, 2017
 *      Author: Jean-Michel Richer
 *
 * This program computes the solutions of the N-Queens problems.
 * Use of ez::arguments, ez::logging and Array of ez::objects
 */


#include "arguments/import.h"
#include "essential/import.h"
#include "logging/import.h"
#include "objects/import.h"

/**
 * Class to solve the NQueens problem based on a very
 * simple implementation that uses three arrays
 * <ul>
 * 	<li>columns: the column where a queen is placed, for
 * 	example columns[3]=7, means that queen 7 is in column
 * 	3</li>
 * 	<li>diagonals1: represent the West to East diagonals</li>
 * 	<li>diagonals2: represent the East to West diagonals</li>
 * </ul>
 * Number of solutions are given by the serie https://oeis.org/A000170.
 * Starting from dimension=4, you should find the following number of
 * solutions: 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596.
 */
class NQueens {
public:
	// dimension of chess board
	integer m_dimension;
	// verbose mode or quiet mode
	boolean m_verbose;
	// logger to print information on file
	ezl::Logger *qlogger;
	// counter of the number of solutions
	long_natural m_nbr_solutions;
	// Timer for the resolution of the problem
	eze::CPUTimer timer;

	ezo::Array<integer> columns;
	ezo::Array<integer> diagonals1;
	ezo::Array<integer> diagonals2;

	/**
	 * constructor
	 * @param dim dimension of chess board
	 * @param verbose verbose mode
	 */
	NQueens(integer dim, boolean verbose) : m_dimension(dim), m_verbose(verbose) {
		columns.resize(Range(1,dim));
		diagonals1.resize(Range(1,2*dim));
		diagonals2.resize(Range(-dim+1,dim-1));
		columns.fill(0);
		diagonals1.fill(0);
		diagonals2.fill(0);
		m_nbr_solutions = 0;
		qlogger = new ezl::FileLogger("queens",
				"queens_logger.txt", ezl::FileLogger::TRUNCATE);
		ezl::LoggerManager::instance().attach(qlogger);
	}

	/**
	 * main method to call after constructor to solve the problem
	 */
	void solve() {
		timer.start();
		// start to solve placing queen 1
		place(1);
		timer.stop();
		(*qlogger) << "start: " << timer << ezl::Logger::endl;
	}

	void place(integer queen) {
		// if we have reach number of queens to place
		// then we have a solution
		if (queen > m_dimension) {
			++m_nbr_solutions;
			if (m_verbose) {
				cout << "solution: " << m_nbr_solutions << endl;
				print_solution(cout);
			}
			return ;
		}

		// otherwise try to place queen in columns 1 to dimension
		for (int col = 1; col <= m_dimension; ++ col) {
			// if we can place queen in a column then
			// continue with next queen
			if (can_place(queen, col)) {
				if (m_verbose) {
					(*qlogger) << "put queen " << queen << " in column " << col << ezl::Logger::endl;
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
	 * Check whether a queen can be put in given row.
	 * For this we verify that the queen won't threaten
	 * a queen that was positioned previously.
	 * @return true if it is possible, false otherwise
	 */
	bool can_place(integer row, integer col) {
		integer value = columns[col] | diagonals1[row + col] | diagonals2[-row+col];
		return (value == 0);
	}

	void print_solution(ostream& out) {
		for (int r = 1; r <= m_dimension; ++r) {
			int pos = columns[r];
			for (int c = 1; c <= m_dimension; ++c) {
				if (c == pos) {
					cout << "Q";
				} else {
					cout << ".";
				}
			}
			cout << endl;
		}
	}
};

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

	// possible command line parameters
	integer dimension = 8;
	boolean quiet = false;

	try {
		parser.add_integer("dimension", 'd', &dimension, "size of board");
		parser.add_flag("quiet", 'q', &quiet, "quiet mode");

		parser.parse();
	} catch(Exception& e) {
		parser.report_error(e);
	}

	cout << "dimension=" << dimension << endl;

	NQueens nqueens(dimension, !quiet);
	nqueens.solve();
	cout << "found " << nqueens.m_nbr_solutions << " solution(s)" << endl;

	return EXIT_SUCCESS;
}


