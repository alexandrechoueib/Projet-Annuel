/*
 * test_range.h
 *
 *  Created on: Apr 29, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "maths/interval.h"
#include <vector>

using namespace std;
using namespace ez;

namespace eze = ez::essential;
namespace ezm = ez::maths;


TEST( TestInterval, construcor ) {

	ezm::Interval<int> i1( 10, 20 );
	EXPECT_EQ( i1.min(), 10 );
	EXPECT_EQ( i1.max(), 20 );

}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}
