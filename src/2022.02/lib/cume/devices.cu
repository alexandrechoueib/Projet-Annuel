#include "devices.h"
 
using namespace ez::cume;

Devices *Devices::_instance = NULL;

Devices::Devices() {
	cume_check( cudaGetDeviceCount(&devices_count) );
	devices = new cudaDeviceProp[ devices_count ];
    for (integer i = 0; i < devices_count; ++i) {
		cume_check( cudaGetDeviceProperties(&devices[i], i) );
    }
}

Devices& Devices::instance() {
	if (_instance == NULL) {
		_instance = new Devices;
	}
	return *_instance;
}
	
Devices::~Devices() {
	delete [] devices;
}
	
void Devices::select(integer device_id) {
	ensure((device_id >= 0) && (device_id < devices_count));
	cume_check( cudaSetDevice(device_id) );
}

cudaDeviceProp *Devices::device_properties(integer device_id) {
	ensure((device_id >= 0) && (device_id < devices_count));
	return &devices[device_id];
}

void Devices::memory_report(ostream& out) {
	size_t free_byte;
    size_t total_byte;
    cume_check( cudaMemGetInfo( &free_byte, &total_byte) );
    double free_db = static_cast<double>(free_byte);
    double total_db = static_cast<double>(total_byte);
    double used_db = total_db - free_db ;
	const double one_mb = 1024.0*1024.0;
    out << "GPU memory usage: used = " << (used_db/one_mb)
            << ", free = " << (free_db/one_mb)
            << ", total = " << (total_db/one_mb) << endl;
}


ostream& Devices::print(ostream& out) {
	for (integer i=0; i<devices_count; ++i) {
		out << "device " << i << ": " << devices[i].name;
		out << " " << devices[i].totalGlobalMem << " bytes";
		natural mem = devices[i].totalGlobalMem / (1024*1024);
		out << " (" << mem << " Mb)";   
		out << endl;
	}
	return out;
}
