/*
 * devices.h
 *
 *  Created on: Jun 19, 2018
 *      Author: richer
 */

#ifndef SRC_VERSION_2018_06_CUME_DEVICES_H_
#define SRC_VERSION_2018_06_CUME_DEVICES_H_

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Definition of a class to handle GPU (device) characteristics
// features
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include <cuda.h>

#include "base.h"

namespace ez {

namespace cume {

/**
 * The Devices class is implemented as a singleton design
 * pattern and helps get information and select the devices
 */
class Devices {
protected:
	/**
	 * number of devices on computer
	 */
	integer devices_count;

	/**
	 * array of information about each device
	 */
	cudaDeviceProp *devices;

	static Devices *_instance;

public:
	enum {
		DEVICE_0 = 0,
		DEVICE_1 = 1,
		DEVICE_2 = 2,
		DEVICE_3 = 3
	};

	static Devices& instance();

	~Devices() ;

	/**
	 * @return number of devices
	 */
	integer count() {
		return devices_count;
	}

	void select(integer device_id);

	cudaDeviceProp *device_properties(integer device_id);

	void memory_report(ostream& out);

    ostream& print(ostream& out);

    /**
     * display information for each device on stream
     */
    friend ostream& operator<<(ostream& out, Devices& obj) {
    	return obj.print(out);
    }

protected:
	Devices();

};

} // end of namespace cume

} // end of namespace ez


#endif /* SRC_VERSION_2018_06_CUME_DEVICES_H_ */
