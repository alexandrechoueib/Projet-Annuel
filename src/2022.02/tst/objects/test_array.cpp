/*
 *
 *  Created on: Apr 29, 2022
 *      Author: abdellah
 */


/*
#include <gtest/gtest.h>
#include "objects/array.h"
#include "essential/range.h"


namespace eze = ez::essential;
namespace ezo = ez::objects;

TEST( TestArray, Constructor ) {
    ezo::Array<int> array(10, 0);
	EXPECT_EQ( array.get(5), 0 );

}

TEST( TestArray, ConstructorRange ) {
	ezo::Array<int> array((new Range(50,60)), 1);
	EXPECT_EQ( array.get(55), 1 );
}

TEST( TestArray, TestConstructorCopy) {
    ezo::Array<int> arraySource(10, 0);
	ezo::Array<int> array(arraySource);
	EXPECT_EQ( array.get(5), 0 );
}

TEST( TestArray, TestSetter ) {
	ezo::Array<int> array(10, 0);
    array.set(2,3);
	EXPECT_EQ( array.get(2), 3 );
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}

*/
