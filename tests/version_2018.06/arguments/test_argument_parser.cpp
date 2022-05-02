/*
 * test_argument_parser.cpp
 *
 *  Created on: May 8, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include <vector>
#include "arguments/import.h"

using namespace std;
using namespace eze;
using namespace eza;

string program_name = "foo.exe";
string program_desc = "test";

TEST(TestArgumentParser, long_option) {
	integer cl_nb_args = 5;
	const char *cl_ar_args[] = {"foo", "--integer=-3333", "--real=3.1415",
			"--natural=4294967295", "--text=abc" };

	integer my_int = 0;
	natural my_nat = 0;
	real    my_real = 0.0;
	text	my_text;

	ArgumentParser parser(program_name, program_desc,
			cl_nb_args, const_cast<char **>(cl_ar_args));
	parser.add_integer("integer", 'i', &my_int, "an integer");
	parser.add_natural("natural", 'n', &my_nat, "a natural");
	parser.add_real("real", 'r', &my_real, "a real");
	parser.add_text("text", 't', &my_text, "a text");

	parser.parse();

	EXPECT_EQ(my_int, -3333);
	EXPECT_FLOAT_EQ(my_real, 3.1415);
	EXPECT_EQ(my_nat, 4294967295);
	EXPECT_EQ(my_text, "abc");
}

TEST(TestArgumentParser, short_option) {
	integer cl_nb_args = 9;
	const char *cl_ar_args[] = {"foo", "-i", "-3333", "-r", "3.1415",
			"-n", "4294967295", "-t", "abc" };

	integer my_int = 0;
	natural my_nat = 0;
	real    my_real = 0.0;
	text	my_text;

	ArgumentParser parser("foo", "my desc", cl_nb_args, const_cast<char **>(cl_ar_args));
	parser.add_integer("integer", 'i', &my_int, "an integer");
	parser.add_natural("natural", 'n', &my_nat, "a natural");
	parser.add_real("real", 'r', &my_real, "a real");
	parser.add_text("text", 't', &my_text, "a text");

	parser.parse();

	EXPECT_EQ(my_int, -3333);
	EXPECT_FLOAT_EQ(my_real, 3.1415);
	EXPECT_EQ(my_nat, 4294967295);
	EXPECT_EQ(my_text, "abc");
}

int main(int argc, char *argv[]) {

	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}



