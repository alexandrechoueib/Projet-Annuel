
/*
#include <gtest/gtest.h>
#include "objects/grid.h"
#include "essential/range.h"

using namespace std;
using namespace ez;

namespace eze = ez::essential;
namespace ezo = ez::objects;


TEST( TestGrid , Constructor ) {
	ezo::Grid<int> grid(10,20, 0);
	EXPECT_EQ( grid.get(5,15), 0 );
}

TEST( TestGrid, ConstructorRange ) {
	ezo::Grid<int> grid((new Range(50,60)), (new Range(80,90)), 1);
	EXPECT_EQ( grid.get(55,82), 1 );
}

TEST( TestGrid, TestConstructorCopy) {
    ezo::Grid<int> gridSource(10,20, 0);
	ezo::Grid<int> grid(gridSource);
	EXPECT_EQ( grid.get(5,15), 0 );
}

TEST( TestGrid, TestSetter ) {
	ezo::Grid<int> grid(10,20, 0);
    grid.set(2,19,3);
	EXPECT_EQ( grid.get(2,19), 3 );
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}
*/