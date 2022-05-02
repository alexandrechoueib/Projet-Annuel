/*
 * test_integer.cpp
 *
 *  Created on: Apr 10, 2017
 *      Author: richer
 */


#include <gtest/gtest.h>
#include "objects/integer.h"
#include <vector>

using namespace std;
namespace eze = ez::essential;
namespace ezo = ez::objects;

TEST(TestInteger, factorial) {

	vector<integer>  v = { 1, 1, 2, 6, 24, 120, 720, 5040 };
	for (eze::integer i = 0; i < static_cast<eze::integer>(v.size()); ++i) {
		EXPECT_EQ(v[i], ezo::Integer::factorial(i));
	}
}


TEST(TestInteger, fibonacci) {

	vector<eze::integer>  v = { 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144};
	for (eze::integer i = 0; i < static_cast<eze::integer>(v.size()); ++i) {
		//cerr << i << " " <<  Integer::fibonacci(i) << endl;
		EXPECT_EQ(v[i], ezo::Integer::fibonacci(i));
	}
}

TEST(TestInteger, is_prime) {

	vector<eze::integer>  v = { 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1};
	for (eze::integer i = 0; i < static_cast<eze::integer>(v.size()); ++i) {
		cerr << i << " " << ezo::Integer::is_prime(i) << endl;
		EXPECT_EQ(v[i], static_cast<integer>(ezo::Integer::is_prime(i)));
	}
}

TEST(TestInteger, input) {
	istringstream iss(" \n\tInteger{ \t 1234 } ");

	ezo::Integer obj1, obj2(1234);
	obj1.input(iss);
	EXPECT_EQ(obj2, obj1);
}

TEST(TestInteger, compute) {
	ezo::Integer x(1), y(2), z(3), t;

	t = x + y * 3 * z;

	EXPECT_EQ(t.value(), 19);
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}


