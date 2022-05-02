/*
 * test_text_utils.cpp
 *
 *  Created on: Apr 9, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include <vector>
#include "essential/text_utils.h"

using namespace std;
using namespace ez;

TEST(TestTextUtils, trim_right) {
	string s_initial = " \t \n  \n\n  this is \t \n a string\n\n \t   ";
	string s_expected = " \t \n  \n\n  this is \t \n a string";

	ez::essential::TextUtils::trim_right(s_initial);

	EXPECT_EQ(s_initial, s_expected);
}


TEST(TestTextUtils, trim_left) {
	string s_initial = " \t \n  \n\n  this is \t \n a string\n\n \t   ";
	string s_expected = "this is \t \n a string\n\n \t   ";

	ez::essential::TextUtils::trim_left(s_initial);

	EXPECT_EQ(s_initial, s_expected);
}

TEST(TestTextUtils, trim) {
	string s_initial = " \t \n  \n\n  this is \t \n a string\n\n \t   ";
	string s_expected = "this is \t \n a string";

	ez::essential::TextUtils::trim(s_initial);

	EXPECT_EQ(s_initial, s_expected);
}

TEST(TestTextUtils, erase) {
	string to_erase = "\nis";
	string s_initial = " \t \n  \n\n  this is \t \n a string\n\n \t   ";
	string s_expected = " \t     th  \t  a trng \t   ";

	ez::essential::TextUtils::erase(s_initial, to_erase);

	//ez::essential::TextUtils::inspect(cout, s_expected);

	EXPECT_EQ(s_initial, s_expected);
}

TEST(TestTextUtils, lower) {
	string s_initial  = " \t \n  \n\n  This Is \t \n A STRing\n\n \t   ";
	string s_expected = " \t \n  \n\n  this is \t \n a string\n\n \t   ";

	ez::essential::TextUtils::lower(s_initial);

	EXPECT_EQ(s_initial, s_expected);
}

TEST(TestTextUtils, upper) {
	string s_initial  = " \t \n  \n\n  This Is \t \n a string\n\n \t   ";
	string s_expected = " \t \n  \n\n  THIS IS \t \n A STRING\n\n \t   ";

	ez::essential::TextUtils::upper(s_initial);

	EXPECT_EQ(s_initial, s_expected);
}

TEST(TestTextUtils, replace) {
	string s_initial  = " \t \n  \n\n  this is \t \n a string\n\n \t   ";
	string s_expected = " \t \n  \n\n  thas as \t \n a strang\n\n \t   ";

	ez::essential::TextUtils::replace_char(s_initial, 'i', 'a');

	EXPECT_EQ(s_initial, s_expected);
}

TEST(TestTextUtils, explode) {
	string s_initial  = " \t \n  \n\n  this is \t \n a string\n\n \t   ";
	vector<string> words;

	ez::essential::TextUtils::explode(s_initial, words);

	EXPECT_EQ(words.size(), 4);
	EXPECT_EQ(words[0], "this");
	EXPECT_EQ(words[1], "is");
	EXPECT_EQ(words[2], "a");
	EXPECT_EQ(words[3], "string");
}

TEST(TestTextUtils, remove_at) {
	string s_initial  = "this is a string";
	vector<ez::essential::integer> pos = { 1,  6, 2 };
	vector<ez::essential::integer> len = { 5, 10, 2 };
	vector<string> s_expected = { "is a string", "is a ", "ia " };

	for (ez::essential::natural i = 0; i < pos.size(); ++i) {
		ez::essential::TextUtils::remove_at(s_initial, pos[i], len[i]);
		EXPECT_EQ(s_initial, s_expected[i]);
	}
}

TEST(TestTextUtils, ends_with) {
	vector<string> s_initial  = { "a.jpg", "toto.jpg", "a.png", "img_1234.jpg" };
	string ending = ".jpg";
	vector<bool> rmflag = { true, true, true, false };
	vector<bool> result = { true, true, false, true };
	vector<string> s_expected = { "a", "toto", "a.png", "img_1234.jpg" };

	for (ez::essential::natural i = 0; i < s_initial.size(); ++i) {
		bool r = ez::essential::TextUtils::ends_with(s_initial[i], ending, rmflag[i]);
		EXPECT_EQ(r, result[i]);
		EXPECT_EQ(s_initial[i], s_expected[i]);
	}
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}


