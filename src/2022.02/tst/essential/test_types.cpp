/*
 * test_types.cpp
 *
 *  Created on: Apr 15, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include <vector>
#include "essential/scalar_types.h"
#include "essential/types.h"
#include "essential/format.h"

using namespace std;
using namespace ez;

namespace eze = ez::essential;

TEST(TestFormat, TextFormat) {
	eze::integer i;
	vector<eze::text> examples = { "a", "aa", "aaa" };
	vector<eze::text> fmt1_expected = {
			"a...................",
			"aa..................",
			"aaa................." };

	vector<eze::text> fmt2_expected = {
			"..........a.........",
			".........aa.........",
			".........aaa........"};

	vector<eze::text> fmt3_expected = {
			"...................a",
			"..................aa",
			".................aaa"};

	i = 0;
	for (auto s : examples) {
		ostringstream oss;
		oss << eze::Format::left(s, 20, '.');
		eze::text result = oss.str();
		//cout <<  result << endl;
		EXPECT_EQ(fmt1_expected[i], result);
		++i;
	}

	i = 0;
	for (auto s : examples) {
		ostringstream oss;
		oss << eze::Format::center(s, 20, '.');
		eze::text result = oss.str();
		//cout <<  result << endl;
		EXPECT_EQ(fmt2_expected[i], result);
		++i;
	}

	i = 0;
	for (auto s : examples) {
		ostringstream oss;
		oss << eze::Format::right(s, 20, '.');
		eze::text result = oss.str();
		//cout <<  result << endl;
		EXPECT_EQ(fmt3_expected[i], result);
		++i;
	}


}

TEST(TestFormat, NumberFormat) {
	eze::integer i;
	vector<eze::integer> examples = { 1, 10, 1023, -1 };

	vector<eze::text> bin_expected = {
			"1b",
			"1010b",
			"11_1111_1111b",
			"1111_1111_1111_1111_1111_1111_1111_1111b"
	};
	vector<eze::text> oct_expected = {
			"1o",
			"12o",
			"1_777o",
			"37_777_777_777o"
	};
	vector<eze::text> hex_expected = {
			"1h",
			"ah",
			"3_ffh",
			"ff_ff_ff_ffh"
	};

	i = 0;
	for (auto s : examples) {
		ostringstream oss;

		// use macro instead of function
		oss.str("");
		oss << ez_to_bin(s);
		EXPECT_EQ(bin_expected[i], oss.str());

		oss.str("");
		oss << eze::Format::oct(static_cast<eze::natural>(s));
		EXPECT_EQ(oct_expected[i], oss.str());

		oss.str("");
		oss << eze::Format::hex(static_cast<eze::natural>(s));
		EXPECT_EQ(hex_expected[i], oss.str());

		++i;
	}
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}






