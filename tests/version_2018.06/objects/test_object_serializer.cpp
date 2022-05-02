/*
 * test_object_serializer.cpp
 *
 *  Created on: Jul 28, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "objects/real.h"
#include "objects/array.h"
#include "objects/vector.h"
#include <vector>

using namespace std;
using namespace ez;

namespace eze = ez::essential;
namespace ezo = ez::objects;

typedef ezo::Vector<ezo::Object *> VectorOfObject;

TEST(TestObjectSerializer, serialize) {

}

TEST(TestObjectSerializer, unserialize) {
}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}




