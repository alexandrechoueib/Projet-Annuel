/*
 * test_range.h
 *
 *  Created on: Apr 29, 2022
 *      Author: alexandre
 */

#include <gtest/gtest.h>
#include "objects/vector.h"

namespace eze = ez::essential;
namespace ezo = ez::objects;

/*
* Test pour création d'un vector et remplissage du vector avec une valeur
*/
TEST( TestVector, Constructor ) {
	ezo::Vector<int> vec ;

}

TEST( TestVector, fill ) {
	ezo::Vector<int> vec ;
	vec.fill(15);
	EXPECT_EQ( vec.get(0) , 15 );
}

TEST( TestVector, setAValue ) {
	Range range(10,25);
	ezo::Vector<std::string> vec(range) ;
	vec.set(11,"bonjour");
	EXPECT_EQ( vec.get(11) , "bonjour" );
}

TEST( TestVector, contrutorRange ) {
	Range range(10,25);
	ezo::Vector<std::string> vec(range) ;
	vec.set(11,"bonjour");
	EXPECT_EQ( vec.get(11) , "bonjour" );
}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}
