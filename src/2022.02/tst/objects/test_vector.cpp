/*
 * test_range.h
 *
 *  Created on: Apr 29, 2022
 *      Author: alexandre
 */

#include <gtest/gtest.h>
#include "objects/vector.h"
#include "essential/range.h"

using namespace std;
using namespace ez;

namespace eze = ez::essential;
namespace ezo = ez::objects;


TEST( TestVector, Constructor ) {

	ezo::Vector vector(range(10,20));
	//EXPECT_EQ( i1.value(), 10 );

}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}
