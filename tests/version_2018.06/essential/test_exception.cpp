/*
 * test_exception.cpp
 *
 *  Created on: Apr 5, 2018
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "essential/exception.h"
#include "essential/ensure.h"
#include "essential/text_utils.h"
#include <vector>

using namespace std;
using namespace ez;

TEST(TestException, notify_level1) {
	try {
		notify("It is an exception");
		ensure(1 == 0);
	} catch(ez::essential::Exception& e) {
		cout << e.what() << endl;
	}
}

TEST(TestException, notify_remedy_level1) {
	try {
		notify("It is an exception," << "but you can avoid it");
		ensure(1 == 0);
	} catch(ez::essential::Exception& e) {
		cout << e.what() << endl;
	}
}

void call_raise_expcetion1() {
	notify("exception - level 1");
}

void call_raise_expcetion2() {
	try {
		call_raise_expcetion1();
	} catch(ez::essential::Exception& e) {
		cout << "what level2=" << e.what() << endl;
		cout << "RETHROW" << endl;
		notify("exception - level 2" << e.what());
	}
}


TEST(TestException, notify_rethrow) {
	try {
		call_raise_expcetion2();
		ensure(1 == 0);
	} catch(ez::essential::Exception& e) {
		cout << "what lastlevel=" << e.what() << endl;
	}
}

TEST(TestException, ensure_is_integer) {
	try {
		ez::essential::text s1 = "-3";
		ez::essential::text s2 = "abc";
		ensure_is_integer(s1);
		ensure_is_integer(s2);
		ensure(1 == 0);
	} catch(ez::essential::Exception& e) {
		string s = e.what();
		string pattern = "s2";
		ez::essential::TextUtils::contains(s, pattern);
	}
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}






