/*
 * cpu_timer.h
 *
 *   Created on: Apr 2, 2015
 *  Modified on: Feb, 2022
 *       Author: Jean-Michel Richer
 */

/*
    EZLib version 2022.02
    Copyright (C) 2019-2022  Jean-Michel Richer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

*/

#ifndef ESSENTIAL_CPU_TIMER_H_
#define ESSENTIAL_CPU_TIMER_H_

#include <stdint.h>
#include <iostream>
#include <math.h>
#include <chrono>
#include <iomanip>

#include "essential/types.h"
using namespace std;

namespace ez {

namespace essential {

/**
  @CLASS
    class used to measure performances of algorithms.
    @list
  	@item use start() method to start timer
  	@item use stop() method to stop<
  	@item use s() method to obtain the number of seconds
  	between start and stop
  	@item use ns(), us() or ms() to obtain respectively number
  	of nano, micro or milli seconds
  	@endlist
 */
class CPUTimer {
public:
	
	typedef std::chrono::time_point<std::chrono::steady_clock> Clocks;

	typedef std::chrono::nanoseconds  nano_sec;
	typedef std::chrono::microseconds micro_sec;
	typedef std::chrono::milliseconds milli_sec;

	// start event
	Clocks _event_start;
	// stop event
	Clocks _event_stop;

	/**
	 @WHAT
	   Default constructor which sets start and stop events
	   to current date and time defined by the clock
	   
	 */
	CPUTimer();

	/**
	 @WHAT
	   Start timer 
	 
	 @HOW
	   We set the start event to current date and time
	   defined by the clock
	   
	 */
	void start();

	/**
	  @WHAT
	    Stop timer 
	 
	  @HOW
	    We set the stop event to current date and time
	    defined by the clock
	    
	 */
	void stop();

	/**
	  @WHAT
	    Return number of nano seconds from start to stop events
	 
	 */
	long_integer ns();

	/**
	 @WHAT
	   Return number of micro seconds from start to stop events
	 
	 */
	long_integer us();

	/**
	 @WHAT
	   Return number of milli seconds from start to stop events
	   
	 */
	long_integer ms();

	/**
	 @WHAT
	   Return number of seconds from start to stop events
	    
	 */
	long_integer s();

	/**
	 @WHAT
	   Return number of seconds since start event but continue
	   to measure time
	   
	 */
	long_integer s_elapsed();

	/**
	 @WHAT
	   Print information about timer in the following 
	   format hh::mm::ss::ms|time_in_ms|time_in_s.
	 
	 @PARAMETERS
	 	@param:out output stream
	 	 
	 @EXAMPLE
	    For example a timer for sleep(2) will print
	    00::00::02::000|2000|2 as follows:
	    @code
	      CPUTimer timer();
	 	  timer.start();
	      sleep(2);
	      timer.stop();
	      cout << timer << endl;
	    @endcode
	 
	 */
	ostream& print( ostream& out );

	/**
	 @WHAT
	   Overloading of output operator
	 
	 @PARAMETERS
	   @param out reference to output stream
	   @param obj reference to a CPUTimer  
	   
	 */
	friend ostream& operator<<( ostream& out, CPUTimer& obj ) {
		return obj.print(out);
	}

private:

	/**
	 @WHAT
	 	return number of clocks
	 */
	Clocks get_clocks();
	
};

} // end of namespace essential

} // end of namespace ez


#endif /* ESSENTIAL_CPU_TIMER_H_ */
