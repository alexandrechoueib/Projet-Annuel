/*
 * test_statistics.cpp
 *
 *  Created on: Apr 10, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "maths/statistics.h"
#include <vector>

using namespace std;
namespace eze = ez::essential;
namespace ezm = ez::maths;

TEST(TestStatistics, percentage) {
	ezm::Series<eze::natural> series;

	const eze::natural max_values = 5;

	for (eze::natural i=1; i<=max_values; ++i) {
		ostringstream oss;
		oss << "v" << i;
		ezm::NaturalValue x(i, oss.str());
		series.record(x);
	}

	cout << series << endl;
	series.percentage();
	cout << series << endl;
	series.export_as_series(cout);
	cout << endl;
}

TEST(TestStatistics, average) {
	ezm::Series<eze::integer> series;

	const eze::natural max_values = 5;

	for (eze::natural i=1; i<=max_values; ++i) {
		ostringstream oss;
		oss << "v" << i;
		ezm::IntegerValue x(-10 + i*2, oss.str());
		series.record(x);
	}

	cout << series << endl;
	ezm::Statistics<eze::integer, eze::real> stats(series);
	stats.compute();
	cout << stats << endl;
	cout << endl;
}

TEST(TestStatistics, stddev) {
	vector<integer> v = { 60, 56, 61, 68, 51, 53, 69, 54 };
	ezm::Series<eze::integer> series(v);

	cout << "series=" << series << endl;

	ezm::Statistics<eze::integer, eze::real> stats(series);
	stats.compute();

	cout << stats << endl;

	EXPECT_EQ(stats.sum(), 472);
	EXPECT_FLOAT_EQ(stats.average(), 59.0);
	EXPECT_FLOAT_EQ(stats.variance(), 40.0);
	EXPECT_FLOAT_EQ(stats.standard_deviation(), 6.3245554);

}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}






