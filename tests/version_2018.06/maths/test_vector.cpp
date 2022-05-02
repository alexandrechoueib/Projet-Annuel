/*
 * test_vector.cpp
 *
 *  Created on: Aug 4, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "essential/import.h"
#include "maths/vector.h"
#include <vector>

using namespace std;
namespace eze = ez::essential;
namespace ezm = ez::maths;

TEST(TestVector, constructors) {
	const eze::natural max_values = 5;
	const eze::real default_value  = 3.5;
	eze::real sum;

	ezm::Vector<eze::real> v1(max_values);

	for (eze::natural i=1; i<=max_values; ++i) {
		v1[i] = default_value;
	}

	sum = std::accumulate(v1.begin(), v1.end(), 0.0);
	EXPECT_EQ(default_value * max_values, sum);

	ezm::Vector<eze::real> v2(v1);
	sum = std::accumulate(v2.begin(), v2.end(), 0.0);
	EXPECT_EQ(default_value * max_values, sum);

	ezm::Vector<eze::real> v3(2);
	v3 = v1;
	sum = std::accumulate(v3.begin(), v3.end(), 0.0);
	EXPECT_EQ(default_value * max_values, sum);

}

TEST(TestVector, remove) {
	const eze::natural max_values = 10;
	eze::real sum;

	ezm::Vector<eze::real> v1(max_values);

	for (eze::natural i=1; i<=max_values; ++i) {
		v1[i] = i;
	}

	v1.remove(3);
	//cout << "v1=" << v1 << endl;
	sum = std::accumulate(v1.begin(), v1.end(), 0.0);
	EXPECT_EQ(v1.size(), 9);
	EXPECT_EQ(1+2+4+5+6+7+8+9+10, sum);

	cout << "!v1=" << v1 << endl;
	v1.remove(1,3);
	cout << "!v1=" << v1 << endl;
	sum = std::accumulate(v1.begin(), v1.end(), 0.0);
	EXPECT_EQ(v1.size(), 6);
	EXPECT_EQ(5+6+7+8+9+10, sum);

	v1.remove(4,10);
	//cout << "v1=" << v1 << endl;
	sum = std::accumulate(v1.begin(), v1.end(), 0.0);
	EXPECT_EQ(v1.size(), 3);
	EXPECT_EQ(5+6+7, sum);



}

TEST(TestVector, add) {
	const eze::natural max_values = 10;

	ezm::Vector<eze::real> v0(max_values);
	ezm::Vector<eze::real> v1(max_values);
	ezm::Vector<eze::real> v2(max_values);
	ezm::Vector<eze::real> v3(max_values);
	ezm::Vector<eze::real> v4;

	std::fill(v0.begin(), v0.end(), 5.0);
	std::fill(v1.begin(), v1.end(), -3.0);
	std::fill(v2.begin(), v2.end(), 7.0);
	std::fill(v3.begin(), v3.end(), 2.0);
	v4 = (v1 * v0 + v2) / v3;
	cout << "v4=" << v4 << endl;

	EXPECT_FLOAT_EQ(v4[1], -4.0);
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}




