/*
 * test_matrix.cpp
 *
 *  Created on: Aug 5, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "essential/import.h"
#include "maths/matrix.h"
#include <vector>

using namespace std;
namespace eze = ez::essential;
namespace ezm = ez::maths;

TEST(TestMatrix, constructors) {
	const eze::natural max_rows = 4;
	const eze::natural max_cols = 6;
	eze::real sum;

	ezm::Matrix<eze::real> m1(max_rows, max_cols);

	m1.fill(1.0);

	sum = std::accumulate(m1.begin(), m1.end(), 0.0);
	EXPECT_EQ(max_rows * max_cols, sum);

	ezm::Matrix<eze::real> m2(m1);
	sum = std::accumulate(m2.begin(), m2.end(), 0.0);
	EXPECT_EQ(max_rows * max_cols, sum);

	ezm::Matrix<eze::real> m3;
	m3 = m1;
	sum = std::accumulate(m3.begin(), m3.end(), 0.0);
	EXPECT_EQ(max_rows * max_cols, sum);

	ezm::Matrix<eze::real> m4(2,3, {1,2,3,4,5,6});
	sum = std::accumulate(m4.begin(), m4.end(), 0.0);
	EXPECT_EQ((6*7)/2, sum);
}

TEST(TestMatrix, operators) {
	const eze::natural max_rows1 = 2;
	const eze::natural max_cols1 = 6;
	const eze::natural max_cols2 = 3;

	ezm::Matrix<eze::real> m1(max_rows1, max_cols1, {
			1, 2, -1, 3, -4, 5,
			-6, 1, 1, -2, 3, 4
	});

	ezm::Matrix<eze::real> m2(max_cols1, max_cols2, {
			1, 2, 3,
			-1, 3, -1,
			-4, 5, -2,
			-6, 1, -3,
			1, -2, 4,
			3, 4, 5
	});

	ezm::Matrix<eze::real> m3(max_rows1, max_cols2, {
			-1, -2, -3,
			-4, -5, -6
	});


	ezm::Matrix<eze::real> m4;
	ezm::Matrix<eze::real> expected_m4(2, 3, {
			5, 32, 0,
			12, -1, 11

	});

	m4 = m1 * m2 + m3;

	cout << "m4=" << m4 << endl;
	EXPECT_EQ(m4, expected_m4);

}

TEST(TestMatrix, product_vector) {
	const eze::natural max_rows1 = 2;
	const eze::natural max_cols1 = 6;

	ezm::Matrix<eze::real> m1(max_rows1, max_cols1, {
			1, 2, -1, 3, -4, 5,
			-6, 1, 1, -2, 3, 4
	});

	ezm::Vector<eze::real> v1(max_cols1, {
			1, 2, 3, 4, 5, 6
	});

	ezm::Vector<eze::real> v2;

	ezm::Vector<eze::real> expected_v2(2, {
			24, 30
	});

	v2 = m1 * v1;
	cout << v2 << endl;
	EXPECT_EQ(v2.size(), 2);
	EXPECT_EQ(v2, expected_v2);

	v2 = v1 * m1;
	cout << v2 << endl;
	EXPECT_EQ(v2.size(), 2);
	EXPECT_EQ(v2, expected_v2);
}

TEST(TestMatrix, transpose) {

	ezm::Matrix<eze::real> m1(4, 4, {
			1, 2, -1, 3,
			-4, 5, -6, 1,
			1, -2, 3, 4,
			0, 0, 0, 2
	});

	m1.transpose();

	ezm::Matrix<eze::real> expected_m1(4, 4, {
			1, -4,  1, 0,
			2,  5, -2, 0,
			-1, -6,  3, 0,
			3,  1,  4, 2
	});

	EXPECT_EQ(m1, expected_m1);
}

TEST(TestMatrix, remove_row) {
	natural rows = 4;
	natural cols = 5;
	ezm::Matrix<eze::real> m1(rows, cols, {
			1,  2, -1, 3, 3,
			2,  5, -6, 1, 3,
			3, -2,  3, 4, 3,
			4,  0,  0, 2, 3
	});

	natural expected_size =  rows * cols;
	while (m1.size() != 0) {
		m1.remove_row(m1.size_y());
		cout << "size = " << m1.size() << endl;
		expected_size -= cols;
		EXPECT_EQ(m1.size(), expected_size);
		cout << "m1=" << m1 << endl;
	}
	EXPECT_EQ(0, m1.size());

}

TEST(TestMatrix, remove_column) {
	natural rows = 4;
	natural cols = 5;
	ezm::Matrix<eze::real> m1(rows, cols, {
			1, 2, 3, 4, 5,
			1, 2, 3, 4, 5,
			1, 2, 3, 4, 5,
			1, 2, 3, 4, 5
	});

	m1.remove_column(3);
	cout << "m1=" << m1 << endl;
	EXPECT_EQ(rows*(cols-1), m1.size());
	EXPECT_EQ(rows, m1.size_y());
	EXPECT_EQ(cols-1, m1.size_x());

}

TEST(TestMatrix, dot) {
	natural rows = 1;
	natural cols = 5;
	ezm::Matrix<eze::real> m1(rows, cols, {
			1, 2, 3, 4, 5
	});
	ezm::Matrix<eze::real> m2(rows, cols, {
			1, 2, 0, 2, 1
	});

	EXPECT_EQ(m1.dot(), 55);
	EXPECT_EQ(m2.dot(), 10);
	cout << "m1=" << m1;
	cout << "m1.dot()=" << m1.dot() << endl;
	cout << "m2=" << m2;
	cout << "m2.dot()=" << m2.dot() << endl;
	m1 -= m2;
	cout << "m1-m2=" << m1;
	EXPECT_EQ(m1.dot(), 29);
}

/**
 * main method to start tests
 */
int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}
