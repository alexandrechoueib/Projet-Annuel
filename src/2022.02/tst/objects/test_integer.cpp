/*
 * test_range.h
 *
 *  Created on: Apr 29, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "objects/integer.h"

using namespace std;
using namespace ez;

namespace eze = ez::essential;
namespace ezo = ez::objects;


TEST( TestInteger, Constructor ) {

	ezo::Integer i1( 10 );
	EXPECT_EQ( i1.value(), 10 );

}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}
