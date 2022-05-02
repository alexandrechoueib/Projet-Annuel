/*
 * test_array.cpp
 *
 *  Created on: May 1, 2017
 *      Author: richer
 */


#include <gtest/gtest.h>
#include "objects/real.h"
#include "objects/array.h"
#include "extensions/import.h"
#include <vector>

using namespace std;
using namespace ez;

namespace eze = ez::essential;
namespace ezo = ez::objects;

typedef ezo::Array<eze::real> FloatArray;
typedef ezo::Array<eze::integer> IntegerArray;

TEST(TestArray, Constructor) {
	FloatArray a1;
	EXPECT_EQ(a1.size(), 0);

	FloatArray a2(1, 10);
	a2.fill(0.5);
	EXPECT_EQ(a2.size(), 10);

	a1 = a2;
	EXPECT_EQ(a1.size(), 10);

	FloatArray a3(Range(1,10), { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

}

TEST(TestArray, access) {
	FloatArray a1(-10, 10);
	a1.fill(0.5);

	for (integer i = -10; i<= 10; ++i) {
		EXPECT_FLOAT_EQ(a1[i], 0.5);
	}

	for (auto it = a1.begin(); it != a1.end(); ++it) {
		EXPECT_FLOAT_EQ(a1[(*it)], 0.5);
	}
}


TEST(TestArray, sum) {
	FloatArray a1(1, 10);
	a1.fill(0.5);
	EXPECT_EQ(ezx::sum(a1, ezo::Real::zero), a1.size() / 2);

	FloatArray a2(1, 10);
	ezx::generate(a2, ezx::Generator<eze::real>());
	EXPECT_EQ(ezx::sum(a2, ezo::Real::zero), (a2.size() * (a2.size() + 1))/ 2);
}


class MyGenerator {
public:
	int seed;

	MyGenerator() {
		seed = 1;
	}

	int operator()() {
		int value = seed;
		seed *= 2;
		return value;
	}
};

/**
 * declare array and fill with function
 */
TEST(TestArray, fill_generator) {

	IntegerArray a(eze::Range(-10,10));

	a.generate(MyGenerator());

	integer sum = ezx::sum(a, ezo::Integer::zero);

	EXPECT_EQ(sum, 2097151);

}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}

