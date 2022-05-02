/*
 * test_volume.cpp
 *
 *  Created on: Aug 14, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "essential/import.h"
#include "extensions/import.h"
#include "maths/matrix.h"
#include "maths/volume.h"
#include <vector>

using namespace std;
namespace eze = ez::essential;
namespace ezm = ez::maths;

TEST(TestVolume, constructors) {
	const eze::natural x = 4;
	const eze::natural y = 6;
	const eze::natural z = 3;
	eze::real sum;

	ezm::Volume<eze::real> v1(z, y, x);

	v1.fill(1.0);

	sum = std::accumulate(v1.begin(), v1.end(), 0.0);
	EXPECT_EQ(x*y*z, sum);

	ezm::Volume<eze::real> v2(v1);
	sum = std::accumulate(v2.begin(), v2.end(), 0.0);
	EXPECT_EQ(x*y*z, sum);

	ezm::Volume<eze::real> v3;
	v3 = v1;
	sum = std::accumulate(v3.begin(), v3.end(), 0.0);
	EXPECT_EQ(x*y*z, sum);

	ezm::Matrix<eze::real> m4(2,3, {1,2,3,4,5,6});
	sum = std::accumulate(m4.begin(), m4.end(), 0.0);
	EXPECT_EQ((6*7)/2, sum);
}

TEST(TestVolume, fill_generate) {
	const eze::natural x = 4;
	const eze::natural y = 6;
	const eze::natural z = 3;
	eze::real sum;

	ezm::Volume<eze::real> v1(z, y, x);
	ezx::fill(v1, 1);

	v1(1,2,3) = 4;
	v1(3,2,1) = 5;

	sum = ezx::sum(v1, 0.0);
	EXPECT_EQ(v1.size()-2+4+5, sum);

}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}


