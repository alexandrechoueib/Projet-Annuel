/*
 * test_csv_file.cpp
 *
 *  Created on: Mar 30, 2018
 *      Author: richer
 */


#include <gtest/gtest.h>
#include "io/csv_file.h"

using namespace std;
using namespace ez;

namespace eze = ez::essential;
namespace ezo = ez::io;

TEST(TestArray, Read) {
	ezo::CSVFile f("data/csv_test.csv");
	f.read();
	for (auto row : f.get_data()) {
		for (auto v : row) {
			std::cout << v << " @ ";
		}
		std::cout << endl;
	}
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}





