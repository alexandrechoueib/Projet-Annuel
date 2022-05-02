/*
 * test_set.cpp
 *
 *  Created on: May 11, 2017
 *      Author: richer
 */


#include <gtest/gtest.h>
#include "objects/integer.h"
#include "objects/natural.h"
#include "objects/real.h"
#include "objects/set.h"
#include <vector>

using namespace std;
using namespace ez;

namespace eze = ez::essential;
namespace ezo = ez::objects;

typedef ezo::Set<eze::real> RealSet;
typedef ezo::Set<eze::natural> NaturalSet;

TEST(TestSet, Constructor) {
	RealSet a1;
	EXPECT_EQ(a1.size(), 0);

	RealSet a2({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
	EXPECT_EQ(a2.size(), 10);

}

TEST(TestSet, access) {
	RealSet a1;
	integer i;

	for (i = -10; i<= 10; ++i) {
		a1.put(i * 0.5);
	}

	i = -10;
	for (auto it = a1.begin(); it != a1.end(); ++it) {
		EXPECT_FLOAT_EQ(*it, i * 0.5);
		++i;
	}
}


TEST(TestSet, sum) {
	NaturalSet a1;

	natural i = 1;

	while (a1.size() != 20) {
		if (ezo::Natural::is_prime(i)) a1.put(i);
		++i;
	}

	EXPECT_EQ(a1.sum(ezo::Natural::zero), 639);

}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}



