/*
 * test_range.h
 *
 *  Created on: Apr 29, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "essential/range.h"
#include <vector>

using namespace std;
using namespace ez;

namespace eze = ez::essential;

void print(std::vector<eze::integer>& v) {
	for (auto e : v) {
		cout << e << ",";
	}
	cout << endl;
}

TEST(TestRange, size) {
	eze::Range r1(1,10);
	EXPECT_EQ(r1.size(), 10);

	eze::Range r2(-10,10);
	EXPECT_EQ(r2.size(), 21);

	eze::Range r3(-3,3,2);
	EXPECT_EQ(r3.size(), 4);

	eze::Range r4(-3,2,2);
	EXPECT_EQ(r4.size(), 3);

	eze::Range r5(-3,2,3);
	EXPECT_EQ(r5.size(), 2);
}

void check_values(eze::Range& r, std::vector<eze::integer>& v) {
	eze::integer i = 0;
	for (auto e : r) {
		EXPECT_EQ(e, v[i]);
		++i;
	}
}

TEST(TestRange, get_values) {
	std::vector<eze::integer> values;
	std::vector<eze::integer> r1_values = {1,2,3,4,5,6,7,8,9,10};
	std::vector<eze::integer> r2_values = {-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10};
	std::vector<eze::integer> r3_values = { -3,-1,1,3 };
	std::vector<eze::integer> r4_values = { -3,-1,1 };
	std::vector<eze::integer> r5_values = { -3,0};

	eze::Range r1(1,10);
	values.clear();
	r1.get_values(values);
	check_values(r1,r1_values);

	eze::Range r2(-10,10);
	values.clear();
	r2.get_values(values);
	check_values(r2,r2_values);

	eze::Range r3(-3,3,2);
	values.clear();
	r3.get_values(values);
	check_values(r3,r3_values);

	eze::Range r4(-3,2,2);
	values.clear();
	r4.get_values(values);
	check_values(r4,r4_values);

	eze::Range r5(-3,2,3);
	values.clear();
	r5.get_values(values);
	check_values(r5,r5_values);

}

TEST(TestRange, includes) {
	eze::integer i;
	eze::boolean b;

	std::vector<eze::integer> r1_values = {-1, 1,2,3,4,5,6,7,8,9,10, 20};
	eze::text r1_includes = "011111111110";

	std::vector<eze::integer> r2_values = {-11, -10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10, 200};
	eze::text r2_includes = "01111111111111111111110";

	std::vector<eze::integer> r3_values = { -3,-1,1,3, -9, -4, 0, 2, 4 };
	eze::text r3_includes = "111100000";

	std::vector<eze::integer> r4_values = { -3,-1,1, -4, -2, 0, 2, 3 };
	eze::text r4_includes = "11100000";

	std::vector<eze::integer> r5_values = { -3,0, -2, -1, 1, 2, 3};
	eze::text r5_includes = "1100000";

	eze::Range r1(1,10);
	eze::Range r2(-10,10);
	eze::Range r3(-3,3,2);
	eze::Range r4(-3,2,2);
	eze::Range r5(-3,2,3);


	i = 0;
	for (auto e : r1_values) {
		b = (r1_includes[i] == '1') ? true : false;
		//cout << "e=" << e << ", b=" << b << ", r" << r1.includes(e) << endl;
		EXPECT_EQ(b, r1.includes(e));
		++i;
	}

	i = 0;
	for (auto e : r2_values) {
		b = (r2_includes[i] == '1') ? true : false;
		//cout << "e=" << e << ", b=" << b << ", r" << r2.includes(e) << endl;
		EXPECT_EQ(b, r2.includes(e));
		++i;
	}

	i = 0;
	for (auto e : r3_values) {
		b = (r3_includes[i] == '1') ? true : false;
		//cout << "e=" << e << ", b=" << b << ", r" << r3.includes(e) << endl;
		EXPECT_EQ(b, r3.includes(e));
		++i;
	}

	i = 0;
	for (auto e : r4_values) {
		b = (r4_includes[i] == '1') ? true : false;
		//cout << "e=" << e << ", b=" << b << ", r" << r4.includes(e) << endl;
		EXPECT_EQ(b, r4.includes(e));
		++i;
	}

	i = 0;
	for (auto e : r5_values) {
		b = (r5_includes[i] == '1') ? true : false;
		//cout << "e=" << e << ", b=" << b << ", r" << r5.includes(e) << endl;
		EXPECT_EQ(b, r5.includes(e));
		++i;
	}

}

TEST(TestRange, RangeIndex) {

	eze::Range r1(1,10);
	eze::Range r2(-10,10);

	for (auto e : r1) {
		EXPECT_EQ(e-1, r1.to_index(e));
	}

	for (auto e : r2) {
		EXPECT_EQ(e+10, r2.to_index(e));
	}

}

TEST(TestRange, intersects) {

	eze::Range r(1,10);
	eze::Range y1(8,9);
	eze::Range y2(8,12);
	eze::Range y3(-4,1);
	eze::Range n1(-4,-1);
	eze::Range n2(11,20);

	EXPECT_TRUE(r.intersects(y1));
	EXPECT_TRUE(r.intersects(y2));
	EXPECT_TRUE(r.intersects(y3));
	EXPECT_TRUE(y1.intersects(r));
	EXPECT_TRUE(y2.intersects(r));
	EXPECT_TRUE(y3.intersects(r));

	EXPECT_FALSE(r.intersects(n1));
	EXPECT_FALSE(r.intersects(n2));
	EXPECT_FALSE(n1.intersects(r));
	EXPECT_FALSE(n2.intersects(r));
}

TEST(TestRange,for_in_by_hand) {
	eze::integer sum = 0;
	eze::integer maxi = 10;

	for (auto e : eze::Range(1,maxi)) {
		sum += e;
	}
	EXPECT_EQ(sum, maxi*(maxi+1)/2);
	//std::cout << "sum=" << sum << std::endl;
}

TEST(TestRange,for_in_macro) {
	eze::integer sum = 0;
	eze::integer maxi = 10;

	for_in(e,1,maxi) {
		sum += e;
	}
	EXPECT_EQ(sum, maxi*(maxi+1)/2);
	//std::cout << "sum=" << sum << std::endl;
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}







