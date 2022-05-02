#include "cume/import.h"
using namespace ezc;

int main() {
	int nbr_gpus = Devices::instance().count();
	for (int i = 0; i < nbr_gpus; ++i) {
		cudaDeviceProp *properties;
		
		properties = Devices::instance().device_properties( i );
		cout << "Device " << i << ": ";
		cout << properties->name << " (cc=";
		cout << properties->major << ".";
		cout << properties->minor << ")" << endl;
	}
	
	//cout << Devices::instance() << <endl;
	
	return EXIT_SUCCESS;
}
