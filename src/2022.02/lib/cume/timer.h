/*
 * gpu_timer.h
 *
 *  Created on: Jun 19, 2018
 *      Author: richer
 */

#ifndef SRC_VERSION_2018_06_CUME_TIMER_H_
#define SRC_VERSION_2018_06_CUME_TIMER_H_

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Simplification of the usage of cudaEvents
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include <cuda.h>
#include <iostream>
using namespace std;
#include <time.h>

namespace ez {

namespace cume {

class Timer {
protected:
	cudaEvent_t t_start, t_stop;

public:
	/**
	 * constructor with no arguments
	 */
	Timer();

	/**
	 * destructor
	 */
	~Timer();

	/**
	 * start timer
	 */
	void start();
	/**
	 * stop timer
	 */
	void stop();

	float elapsed();
	
	/**
	 * print timer difference in milliseconds
	 */
	ostream& print(ostream& out);

	friend ostream& operator<<(ostream& out, Timer& obj) {
		return obj.print(out);
	}

};

} // end of namespace cume

} // end of namespace ez

#endif /* SRC_VERSION_2018_06_CUME_TIMER_H_ */
