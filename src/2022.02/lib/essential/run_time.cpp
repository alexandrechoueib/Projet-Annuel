/*
 * run_time.cpp
 *
 *  Created on: Apr 21, 2015
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "essential/run_time.h"

#include <signal.h>

using namespace ez::essential;

bool RunTime::s_interrupt_program = false;


void signal_intercept( int signal_no ) {

	RunTime::set_interrupt();
	
}


void RunTime::set_interrupt() {

	s_interrupt_program = true;
	
}


bool RunTime::get_interrupt() {

	return s_interrupt_program;
	
}


void RunTime::init() {

	signal(SIGABRT, signal_intercept);
	signal(SIGILL, signal_intercept);
	signal(SIGHUP, signal_intercept);
	signal(SIGINT, signal_intercept);
	signal(SIGTERM, signal_intercept);

}
