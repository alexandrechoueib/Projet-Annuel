/*
 * test_matrix2d.cpp
 *
 *  Created on: Aug 2, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "objects/integer.h"
#include "objects/matrix2d.h"
#include "objects/array.h"
#include <vector>
#include <numeric>

using namespace std;
namespace eze = ez::essential;
namespace ezo = ez::objects;

TEST(TestMatrix2D, build_fill) {
	ezo::Matrix2D<integer> m(eze::Range(1,6), eze::Range(1,4));

	std::fill(m.begin(), m.end(), 1);

	for (auto y : m.y_range()) {
		for (auto x : m.x_range()) {
			cout << "x=" << x << ",y=" << y << "=> " << m.get(x,y) << endl;
			EXPECT_EQ(1, m.get(x, y));
		}
	}
	cout << m << endl;

	integer sum = std::accumulate(m.begin(), m.end(), 0);
	cout << "sum=" << sum << endl;
	EXPECT_EQ(sum, m.size());
}

TEST(TestMatrix2D, build_fill_column) {
	ezo::Matrix2D<integer> m(eze::Range(1,6), eze::Range(1,4));

	std::fill(m.begin(), m.end(), 1);

	m.fill_column(2, -1);
	cout << m << endl;

	integer sum = std::accumulate(m.begin(), m.end(), 0);
	cout << "sum=" << sum << endl;
	EXPECT_EQ(sum, (m.x_range().size() - 2) * m.y_range().size());
}

TEST(TestMatrix2D, build_fill_row) {
	ezo::Matrix2D<integer> m(eze::Range(1,6), eze::Range(1,4));

	std::fill(m.begin(), m.end(), 1);

	m.fill_row(2, -1);
	cout << m << endl;

	integer sum = std::accumulate(m.begin(), m.end(), 0);
	cout << "sum=" << sum << endl;
	cout << "m.x_range().size()=" << m.x_range().size() << endl;
	EXPECT_EQ(sum, (m.y_range().size() - 2) * m.x_range().size());
}

TEST(TestMatrix2D, build_fill_get_column) {
	ezo::Matrix2D<integer> m(eze::Range(1,6), eze::Range(1,4));

	std::fill(m.begin(), m.end(), 1);

	m.fill_column(2, -1);
	cout << m << endl;

	ezo::Array<int> a;
	m.get_column(2, a);
	cout << a << endl;
	integer sum = std::accumulate(a.begin(), a.end(), 0);
	cout << "sum=" << sum << endl;
	EXPECT_EQ(sum, -m.y_range().size());
}

TEST(TestMatrix2D, build_fill_get_row) {
	ezo::Matrix2D<integer> m(eze::Range(1,6), eze::Range(1,4));

	std::fill(m.begin(), m.end(), 1);

	m.fill_row(2, -1);
	cout << m << endl;

	ezo::Array<int> a;
	m.get_row(2, a);
	cout << a << endl;
	integer sum = std::accumulate(a.begin(), a.end(), 0);
	cout << "sum=" << sum << endl;
	EXPECT_EQ(sum, -static_cast<integer>(m.x_range().size()));
}

TEST(TestMatrix2D, build_fill_transpose) {
	ezo::Matrix2D<integer> m(eze::Range(1,4), eze::Range(1,4),
			{1, 2, 3, 4, -1, -2, -3, -4, 5, 6, 7, 8, -5, -6, -7, -8});
	ezo::Matrix2D<integer> tm(eze::Range(1,4), eze::Range(1,4),
			{1, -1, 5, -5, 2, -2, 6, -6, 3, -3, 7, -7, 4, -4, 8, -8});

	cout << m << endl;

	m.transpose();
	cout << m << endl;

	EXPECT_EQ(m == tm, true);
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}





