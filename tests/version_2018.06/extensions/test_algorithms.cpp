/*
 * test_algorithms.cpp
 *
 *  Created on: Aug 7, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "objects/vector.h"
#include "extensions/import.h"
#include <vector>
#include "essential/import.h"

using namespace std;
using namespace ez;


namespace ezo = ez::objects;
namespace ezx = ez::extensions;

ostringstream oss;

void print_square(integer x) {
	oss << x*x << " " ;
}

integer square(integer x) {
	return x*x;
}

/**
 * Test count and count_if
 */
TEST(TestAlgorithms, count_count_if) {
	natural nbr;

	ezo::Vector<eze::real> v(10);
	ezx::generate(v, ezx::Generator<real>());
	v.put_last(1.0);
	v.put_last(2.0);
	v.put_last(3.0);
	v.put_last(1.0);
	v.put_last(1.0);

	nbr = ez::extensions::count(v, 1.0);
	//cout << v << endl;
	EXPECT_EQ(4, nbr);

	nbr = ez::extensions::count_if(v, [](real x) {
		return x >= 4.0;
	});
	//cout << "sum=" << sum << endl;

	EXPECT_EQ(7, nbr);
}




TEST(TestAlgorithms, for_each) {
	ezo::Vector<eze::integer> v(10);
	ezx::generate(v, ezx::Generator<integer>());
	ez::extensions::for_each(v, print_square);

	string expected_result="1 4 9 16 25 36 49 64 81 100 ";
	EXPECT_EQ(expected_result, oss.str());
	//cout << v << endl;

	ez::extensions::transform(v, square);
	//cout << v << endl;

	integer sum = ez::extensions::sum(v,0);
	//cout << "sum=" << sum << endl;

	EXPECT_EQ(385, sum);
}

TEST(TestAlgorithms, all_diff) {
	ezo::Vector<eze::integer> v1(10);
	ezx::generate(v1, ezx::Generator<integer>());
	ezo::Vector<eze::integer> v2(10);
	ezx::generate(v2, ezx::Generator<integer>());
	v2.put_at(4,7);

	bool b = ezx::all_diff(v1);
	EXPECT_TRUE(b);

	b = ezx::all_diff(v2);
	EXPECT_FALSE(b);

	auto it = ezx::all_diff_pos(v2);
	// position should be 4 in the EZL system but as
	// we are using the STL, it is 4-3 because the STL
	// starts with index 0 and EZL starts with index 1
	EXPECT_EQ(it-v2.begin(), 3);
}

TEST(TestAlgorithms, all_match_simple) {
	ezo::Vector<eze::integer> v1(10);
	ezx::generate(v1, ezx::Generator<integer>());
	ezo::Vector<eze::integer> v2(10);
	ezx::generate(v2, ezx::Generator<integer>());

	ezo::Vector<eze::integer> v3 = v2;
	ezo::Vector<eze::integer> v4 = v2;
	v4.put_at(5,-1);

	bool b;
	b = ezx::all_match(v1, v2);
	EXPECT_TRUE(b);
	b = ezx::all_match(v1, v3);
	EXPECT_TRUE(b);
	b = ezx::all_match(v1, v4);
	EXPECT_FALSE(b);
}

class A : public ezo::Object {
public:
	int m_data;

	A() : ezo::Object(), m_data(0) { }
	A(int v) : ezo::Object(),m_data(v) { }
	~A() { }

	int data() { return m_data; }

	integer compare(const Object& y) {
		A& y_obj = *dynamic_cast<A *>(&const_cast<Object&>(y));
		return m_data - y_obj.m_data;
	}
};

TEST(TestAlgorithms, all_match_class) {
	ezo::Vector<A> v1, v2;
	v1.put_last(A(1));
	v1.put_last(A(2));
	v1.put_last(A(3));
	v2.put_last(A(1));
	v2.put_last(A(2));
	v2.put_last(A(3));


	ezo::Vector<A> v3 = v2;
	ezo::Vector<A> v4 = v2;
	v4.put_at(2,-1);

	bool b;
	b = ezx::all_match(v1, v2);
	EXPECT_TRUE(b);
	b = ezx::all_match(v1, v3);
	EXPECT_TRUE(b);
	b = ezx::all_match(v1, v4);
	EXPECT_FALSE(b);
}

bool comparePtrToA(A *x, A *y) {
	return x->m_data - y->m_data;
}

TEST(TestAlgorithms, all_match_ptr_to_class) {
	ezo::Vector<A *> v1, v2;
	v1.put_last(new A(1));
	v1.put_last(new A(2));
	v1.put_last(new A(3));
	v2.put_last(new A(1));
	v2.put_last(new A(2));
	v2.put_last(new A(3));


	ezo::Vector<A *> v3 = v2;
	ezo::Vector<A *> v4 = v2;
	v4.put_at(2,new A(-1));

	bool b;
	b = ezx::all_match(v1, v2, comparePtrToA);
	EXPECT_TRUE(b);
	b = ezx::all_match(v1, v3, comparePtrToA);
	EXPECT_TRUE(b);
	b = ezx::all_match(v1, v4, comparePtrToA);
	EXPECT_FALSE(b);
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}






