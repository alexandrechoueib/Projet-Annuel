
#include <gtest/gtest.h>
#include "objects/grid.h"
#include "essential/range.h"

using namespace std;
using namespace ez;

namespace eze = ez::essential;
namespace ezo = ez::objects;


TEST( TestGridInitValueBasic, Constructor ) {
	ezo::Grid grid(10,20, 0);
	EXPECT_EQ( grid.get(5,15), 0 );
}

TEST( TestGridInitValueRange, Constructor ) {
	ezo::Grid grid(range(50,60), range(80,90), 1);
	EXPECT_EQ( grid.get(55,82), 1 );
}

TEST( TestGridInitValueCopy, Constructor ) {
    ezo::Grid gridSource(10,20, 0);
	ezo::Grid grid(gridSource);
	EXPECT_EQ( grid.get(5,15), 0 );
}

TEST( TestGridSetter, Constructor ) {
	ezo::Grid grid(10,20, 0);
    grid.set(2,19,3);
	EXPECT_EQ( grid.get(2,19), 3 );
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}
