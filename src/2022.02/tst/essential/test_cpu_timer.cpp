/*
 * test_cpu_timer.cpp
 *
 *  Created on: Apr 8, 2017
 * Modified on: Oct, 2022
 *      Author: Jean-Michel Richer
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

	EXPECT_EQ(timer.ms(), 1000);
	//(timer);
	cout << timer << endl;
}

TEST(TestCPUTimer, seconds_elapsed) {
	ez::essential::CPUTimer timer;

	timer.start();
	for (int i=0; i<5; ++i) {
		sleep(1);
		cout << timer.s_elapsed() << endl;
	}
	timer.stop();

	EXPECT_EQ(timer.s(), 5);
	//(timer);
	cout << timer << endl;
}



int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}



