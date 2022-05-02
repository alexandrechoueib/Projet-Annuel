/*
 * test_mapping.cpp
 *
 *  Created on: Jul 30, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "objects/real.h"
//#include "objects/vector.h"
#include "objects/mapping.h"
#include "objects/integer.h"
#include "objects/real.h"
#include <vector>

using namespace std;
using namespace ez;

namespace eze = ez::essential;
namespace ezo = ez::objects;


TEST(TestMapping, Constructor) {
	ezo::Mapping<text, integer> m1;

	m1.put("un", 1);
	m1.put("deux", 2);
	m1.put("trois", 3);
	m1.put("quatre", 4);
	m1.put("cinq", 5);

	EXPECT_EQ(m1.size(), 5);

}


TEST(TestMapping, _Vector) {
	ezo::Mapping<text, integer> m1;

	m1.put("un", 1);
	m1.put("deux", 2);
	m1.put("trois", 3);
	m1.put("quatre", 4);
	m1.put("cinq", 5);

	cout << m1 << endl;

	ezo::Vector<ezo::Couple<text, integer>> v1;

	m1.to_vector(v1);

	EXPECT_EQ(v1.size(), 5);

	cout << v1 << endl;

}



int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}







