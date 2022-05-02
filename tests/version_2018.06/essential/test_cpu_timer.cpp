/*
 * test_cpu_timer.cpp
 *
 *  Created on: Apr 8, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "essential/cpu_timer.h"
#include <vector>

using namespace std;
using namespace ez;

TEST(TestCPUTimer, compare) {
	ez::essential::CPUTimer timer;

	timer.start();
	sleep(1);
	timer.stop();

	EXPECT_EQ(timer.get_milli_seconds(), 1000);
	//(timer);
	cout << timer << endl;
}

TEST(TestCPUTimer, seconds_elapsed) {
	ez::essential::CPUTimer timer;

	timer.start();
	for (int i=0; i<5; ++i) {
		sleep(1);
		cout << timer.get_seconds_elapsed() << endl;
	}
	timer.stop();

	EXPECT_EQ(timer.get_seconds(), 5);
	//(timer);
	cout << timer << endl;
}



int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}



