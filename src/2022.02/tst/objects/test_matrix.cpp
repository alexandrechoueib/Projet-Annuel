#include <gtest/gtest.h>
#include "objects/matrix.h"

namespace eze = ez::essential;
namespace ezo = ez::objects;

/*
* Test pour création d'un vector et remplissage du vector avec une valeur
*/
TEST( TestVector, Constructor ) {
	ezo::Matrix<int> matrix ;

}

TEST( TestVector, fill ) {
    unsigned int testValue = 15;
	ezo::Matrix<int> matrix ;
	matrix.fill(testValue);
    for(int i=0 ; i < matrix.size_row() ; i++){
        for(int j=0; j < matrix.size_col() ; j++){
	         EXPECT_EQ( matrix.get(i,j) , testValue );
        }
    }
}

TEST( TestVector, setAValue ) {
	Range range_x(10,25);
    Range range_y(5,15);

	ezo::Matrix<std::string> matrix(range_x,range_y) ;
	matrix.set(11,6,"bonjour");
	EXPECT_EQ( matrix.get(11,6) , "bonjour" );
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
