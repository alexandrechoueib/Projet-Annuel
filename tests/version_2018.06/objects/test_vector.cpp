/*
 * test_vector.cpp
 *
 *  Created on: May 5, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "objects/real.h"
#include "objects/vector.h"
#include "objects/integer.h"
#include "objects/real.h"
#include "extensions/import.h"
#include <vector>

using namespace std;
using namespace ez;

namespace eze = ez::essential;
namespace ezo = ez::objects;


TEST(TestVector, Constructor) {
	ezo::Vector<integer> v1(100);
	EXPECT_EQ(v1.size(), 100);

	ezo::Vector<real> v2({ 3.1, 4.2, 5.3, 6.4, 5.5 });
	EXPECT_EQ(v2.size(), 5);

}

class IntegerIotaGenerator {
public:
	integer value;

	IntegerIotaGenerator() {
		value = 1;
	}

	integer operator()() {
		return value++;
	}
};

TEST(TestVector, sum) {
	ezo::Vector<integer> v1(100);
	ezx::fill(v1,2);
	EXPECT_EQ(ezx::sum(v1, ezo::Integer::zero), 200);

	ezo::Vector<real> v2(10);
	ezx::generate(v2, IntegerIotaGenerator());
	EXPECT_EQ(ezx::sum(v2, ezo::Real::zero), 55);

}

TEST(TestVector, put_get_remove) {
	const integer one_hundred = 100;
	ezo::Vector<integer> v1(one_hundred);
	ezx::fill(v1, 1);

	v1.put_first(-10);
	EXPECT_EQ(v1[1], -10);

	v1.put_last(+10);
	EXPECT_EQ(v1.last(), 10);

	EXPECT_EQ(v1.size(), one_hundred + 2);
	EXPECT_EQ(ezx::sum(v1, ezo::Real::zero), one_hundred);

	cout << v1 << endl;

	v1.put_at(3, -7);
	cout << v1 << endl;

	v1.put_at(-3, -8);
	cout << v1 << endl;
}

TEST(TestVector, clone) {
	const integer one_hundred = 100;
	ezo::Vector<integer> v1(one_hundred);
	ezx::generate(v1, IntegerIotaGenerator());

	const integer sum_expected = (one_hundred * (one_hundred + 1)) / 2;
	EXPECT_EQ(ezx::sum(v1, ezo::Real::zero), sum_expected);

	ezo::Vector<integer> *v2 = dynamic_cast<ezo::Vector<integer> *>(v1.clone());
	EXPECT_EQ(ezx::sum(*v2, ezo::Real::zero), sum_expected);

	EXPECT_EQ(v2->compare(v1), 0);
}



int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}




