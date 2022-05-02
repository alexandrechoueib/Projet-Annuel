/*
 * test_logging_manager.cpp
 *
 *  Created on: May 8, 2017
 *      Author: richer
 */


#include <gtest/gtest.h>
#include "logging/logger_manager.h"


using namespace std;
using namespace ez;

TEST(TestLoggerManager, def) {
	ez::logging::Logger *my_log = new ez::logging::ConsoleLogger("my_log");
	ez::logging::LoggerManager::instance().attach(my_log);
	//EXPECT_EQ(timer.get_milli_seconds(), 1000);

}

TEST(TestLoggerManager, def_and_create) {
	ez::logging::Logger& my_log = ez::logging::LoggerManager::instance().get(
		"new_log",
		"file,file_log.log,truncate"
	);
	my_log << "hello world!" << ez::logging::Logger::endl;

}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}


