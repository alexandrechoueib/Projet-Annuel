/*
 * test_format.cpp
 *
 *  Created on: Apr 9, 2018
 *      Author: richer
 */

#include <gtest/gtest.h>
#include <vector>
#include "essential/scalar_types.h"
#include "essential/numeric_base.h"
#include "essential/format.h"


using namespace std;
using namespace ez;

namespace eze = ez::essential;

TEST(TestNumericBase, convert) {
	eze::integer i;
	eze::NumericBase& b_bin = eze::NumericBaseManager::get_instance().by_name("binary");
	eze::NumericBase& b_oct = eze::NumericBaseManager::get_instance().by_name("octal");
	eze::NumericBase& b_hex = eze::NumericBaseManager::get_instance().by_name("hexadecimal");

	std::vector<eze::integer> values = { 1, 2, 3, 2222, -1, -2 };
	std::vector<eze::text> bin_expected = {
			"1",
			"10",
			"11",
			"100010101110",
			"11111111111111111111111111111111",
			"11111111111111111111111111111110"
	};
	std::vector<eze::text> oct_expected = {
			"1",
			"2",
			"3",
			"4256",
			"37777777777",
			"37777777776"
	};
	std::vector<eze::text> hex_expected = {
			"1",
			"2",
			"3",
			"8ae",
			"ffffffff",
			"fffffffe"
	};


	i = 0;
	for (auto e : values) {
		eze::text result = b_bin.convert(e);
		//cout << e << ": " << result << endl;
		EXPECT_EQ(bin_expected[i], result);
		++i;
	}

	i = 0;
	for (auto e : values) {
		eze::text result = b_oct.convert(e);
		//cout << e << ": " << result << endl;
		EXPECT_EQ(oct_expected[i], result);
		++i;
	}

	i = 0;
	for (auto e : values) {
		eze::text result = b_hex.convert(e);
		//cout << e << ": " << result << endl;
		EXPECT_EQ(hex_expected[i], result);
		++i;
	}

}

TEST(TestNumericBase, by) {

	eze::NumericBase& b_bin = eze::NumericBaseManager::get_instance().by_name("binary");
	EXPECT_EQ(b_bin.name, "binary");

	eze::NumericBase& b_oct = eze::NumericBaseManager::get_instance().by_suffix("123o");
	EXPECT_EQ(b_oct.name, "octal");

	eze::NumericBase& b_hex = eze::NumericBaseManager::get_instance().by_suffix("abcdef_h");
	EXPECT_EQ(b_hex.name, "hexadecimal");

	eze::NumericBase& b_dec = eze::NumericBaseManager::get_instance().by_suffix("1234567");
	EXPECT_EQ(b_dec.name, "decimal");

}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}






