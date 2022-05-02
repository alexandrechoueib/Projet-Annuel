#include "kernel.h"

using namespace ez::cume;

Kernel::Kernel(natural nbt, integer dev_id) {
	grid.x = grid.y = grid.z = 1;
	block.x = block.y = block.z = 1;
	shared = 0;
	stream = 0;
	required_threads = nbt;
		
	device_id = dev_id;
	if ((dev_id < 0) || (dev_id >= Devices::instance().count())) {
		notify("Device of id " << dev_id << " is not allowed" << endl << "Maximum number of devices is " << Devices::instance().count());
	}
	
	// create resource in host and device memory
	cpu_resource = new Resource;
	cpu_resource->kernel_type = KERNEL_TYPE_NONE;
	//gpu_resource = cume_malloc<Resource>(1);
	cume_new_var(gpu_resource, Resource);
	cume_push(gpu_resource, cpu_resource, Resource, 1);
	
} 

Kernel::Kernel(const Kernel& obj) {
	grid = obj.grid;
	block = obj.block;
	shared = obj.shared;
	stream = obj.stream;
	required_threads = obj.required_threads;
	cpu_resource = new Resource;
	cpu_resource->kernel_type = obj.cpu_resource->kernel_type;
	//gpu_resource = cume_malloc<Resource>(1);
	cume_new_var(gpu_resource, Resource);
	cume_push(gpu_resource, cpu_resource, Resource, 1);
	device_id = obj.device_id;
}

/**
 * assignment operator overloading
 */
Kernel& Kernel::operator=(const Kernel& obj) {
	if (&obj != this) {
		grid = obj.grid;
		block = obj.block;
		shared = obj.shared;
		stream = obj.stream;
		required_threads = obj.required_threads;
		cpu_resource->kernel_type = obj.cpu_resource->kernel_type;
		cume_push(gpu_resource, cpu_resource, Resource, 1);
		device_id = obj.device_id;
	}
	return *this;
}


/**
 * destructor
 */
Kernel::~Kernel() {
	delete cpu_resource;
	cume_free(gpu_resource);
}


/**
 * function to call after construction of an instance
 * of KernelConfig in order to setup the grid and block
 * dimensions. First the grid and block mode/type are
 * set followed by the dimensions
 * @param gty grid type (X, XY, XYZ or GUESS)
 * @param gty grid type (X, XY or XYZ)
 * @param dimensions
 * For example to have a grid of 4 blocks, each block
 * having 32 by 8 threads, use :
 * set_config(KernelConfig::GRID_X, KernelConfig::GRID_XY, 4, 32, 8)
 * then grid.x = 4, block.x = 32, block.y = 8
 *
 * Use GRID_GUESS to let the method determine how many blocks are
 * required and provide only information for the block:
 * KernelConfig kcfg(1024);
 * kcfg.set_config(KernelConfig::GRID_GUESS, KernelConfig::GRID_X, 128)
 * then grid.x = 1024/128 = 8
 */
void Kernel::configure(integer gmod, integer bmod, ...) {
	va_list vl;
	va_start(vl, bmod);
	
	switch(gmod) {
		case GRID_1:
			grid.x = grid.y = grid.z = 1;
			break;
		case GRID_X:
			grid.x = va_arg(vl, int);
			break;
		case GRID_XY:
			grid.x = va_arg(vl, int);
			grid.y = va_arg(vl, int);
			break;
		case GRID_XYZ:
			grid.x = va_arg(vl, int);
			grid.y = va_arg(vl, int);
			grid.z = va_arg(vl, int);
			break;	
		default:
			break;	
	};
	
	switch(bmod) {
		case BLOCK_1:
			block.x = block.y = block.z = 1;
			break;
		case BLOCK_X:
			block.x = va_arg(vl, int);
			break;
		case BLOCK_XY:
			block.x = va_arg(vl, int);
			block.y = va_arg(vl, int);
			break;
		case BLOCK_XYZ:
			block.x = va_arg(vl, int);
			block.y = va_arg(vl, int);
			block.z = va_arg(vl, int);
			break;	
			
	};
	
	if ((gmod & GRID_GUESS) != 0) {
		if ((gmod & GRID_X) != 0) {
			integer nbr_threads_in_block = block.x * block.y * block.z;
			grid.x = (required_threads + nbr_threads_in_block - 1) / nbr_threads_in_block;
			grid.y = grid.z = 1;
			gmod = GRID_X;
		} else if ((gmod & GRID_XY) != 0) {
			integer nbr_threads_in_block = block.x * block.y * block.z;
			integer nbr_blocks = required_threads / nbr_threads_in_block;
			cerr << "nbr_blocks = " << nbr_blocks << endl;
			cerr << "nbr_threads_in_block = " << nbr_threads_in_block << endl;
			integer k = floor(sqrt(static_cast<double>(nbr_blocks)));
			while (k*k < nbr_blocks) ++k;
			grid.x = grid.y = k;
			grid.z = 1;
			gmod = GRID_XY;
			cout << "k = " << k << endl;
		} else {
			notify("GRID_GUESS not defined for GRID_XYZ");
		}	
	}
	
	cpu_resource->kernel_type = static_cast<KernelType_t>(gmod * 100 + bmod);
	//cout << "r = " << cpu_resource->kernel_type << endl;
	cume_push(gpu_resource, cpu_resource, Resource, 1);
	
	// check if size of block fits maxThreadsPerBlock
	// constraint
	natural total_threads_in_block = block.x * block.y * block.z;
	
	cudaDeviceProp *prop = Devices::instance().device_properties(device_id); 
	integer max_threads_per_block = prop->maxThreadsPerBlock;
	 
	if (total_threads_in_block > max_threads_per_block) {
		notify("maximum number of threads per block on device " << device_id
			<< " is set to " << max_threads_per_block << endl
			<< "and you have requested " << total_threads_in_block << endl);
	}
	
	if ((gmod & GRID_NO_CHECK) == 0) {
		// check that number of threads is greater or equal
		// to number of threads required
		natural total_threads = grid.x * grid.y * grid.z;
		total_threads *= block.x * block.y * block.z;
	
		if (total_threads < required_threads) {
			ostringstream oss;
		
			notify("Error: number of threads of grid and block ("
				<< total_threads << ")" 
				<< " is less than required number of threads ("
				<< required_threads <<  ")"); 
		}
	}
}


ostream& Kernel::print(ostream& out) {
	out << "block = (" << block.x << ", " << block.y << ", " << block.z << ")" << endl;
	out << "grid = (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << endl;
	natural total = block.x * block.y * block.z;
	total *= grid.x * grid.y * grid.z;
	out << "threads of grid and block = " << total << endl; 
	out << "required threads = " << required_threads << endl;
	out << "shared = " << shared << endl;
	out << "stream = " << stream << endl;
	
	return out;
}
