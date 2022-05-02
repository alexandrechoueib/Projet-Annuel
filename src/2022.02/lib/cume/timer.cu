#include "timer.h"

using namespace ez::cume;

Timer::Timer() {
	cudaEventCreate(&t_start);
	cudaEventCreate(&t_stop);
}

/**
 * destructor
 */
Timer::~Timer() {
	cudaEventDestroy(t_start);
	cudaEventDestroy(t_stop);
}

/**
 * start timer
 */
void Timer::start() {
	cudaEventRecord(t_start, 0);
}

/**
 * stop timer
 */
void Timer::stop() {
	cudaEventRecord(t_stop, 0);
	cudaEventSynchronize(t_stop);
}	

float Timer::elapsed() {
	float milli_seconds = 0.0f;
	cudaEventElapsedTime(&milli_seconds, t_start, t_stop);
	return milli_seconds;
}

/**
 * print timer difference in milliseconds
 */
ostream& Timer::print(ostream& out) {
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, t_start, t_stop);
	out.setf(ios::fixed);
	out.precision(2);
	out << elapsed_time << "ms";
	return out;
}
