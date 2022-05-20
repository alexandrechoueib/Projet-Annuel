
#include <gtest/gtest.h>
#include "objects/mesh.h"
#include "essential/range.h"

using namespace std;
using namespace ez;

namespace eze = ez::essential;
namespace ezo = ez::objects;


TEST( TestMeshInitValueBasic, Constructor ) {
	ezo::Mesh mesh(10,20,30, 0);
	EXPECT_EQ( mesh.get(5,15,25), 0 );
}

TEST( TestMeshInitValueRange, Constructor ) {
	ezo::Mesh mesh(range(50,60), range(80,90), range(20,30), 1);
	EXPECT_EQ( mesh.get(55,82,28), 1 );
}

TEST( TestMeshInitValueCopy, Constructor ) {
    ezo::Mesh meshSource(10,20,30, 0);
	ezo::Mesh mesh(meshSource);
	EXPECT_EQ( mesh.get(5,15,24), 0 );
}

TEST( TestMeshSetter, Constructor ) {
	ezo::Mesh mesh(10,20,30, 0);
    mesh.set(2,19,23,3);
	EXPECT_EQ( mesh.get(2,19,23), 3 );
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}
