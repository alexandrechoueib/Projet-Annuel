#include "essential/import.h"
#include "arguments/import.h"
#include <cuComplex.h>
using namespace std;
#include <getopt.h>
#include "cume/import.h"
#include "complex.h"

// ==================================================================
// global variables
// ==================================================================
int device_id = 0;
int block_size = 16;

int image_width = 1024;
int image_height = 1024;
bool view_flag = false;
int iterations = 200;
REAL constant_real = 0.3;
REAL constant_imag = 0.5;

const REAL xmin = -2.4;
const REAL xmax = 2.4;
const REAL ymin = -1.5;
const REAL ymax = 1.5;
const REAL NORM_THRESOLD = 4.0;

/**
 * determine if given point is in Julia Set
 *
 */
__device__ REAL julia(int x, int y, int IMG_W, int IMG_H, const int max_iter,
		const REAL cr, const REAL ci) {

	aComplex z(x * (REAL)(xmax - xmin)/IMG_W + xmin,
			y * (REAL)(ymax - ymin)/IMG_H + ymin);
	aComplex c(cr, ci);

	for (int k=0; k<max_iter; k++) {
		z = z * z + c;
		if (z.norm() > NORM_THRESOLD) {
			return NORM_THRESOLD;
		}
	}
	return z.norm();
}

/**
 * compute julia set
 */
__global__ void kernel(unsigned char *ptr, int IMG_W, int IMG_H, const int max_iter,
		const REAL cr, const REAL ci) {
	// map from threadIdx/BlockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * gridDim.x * blockDim.x;

	if (offset < IMG_W * IMG_H) {
		// now calculate the value at that position
		REAL juliaValue = julia(x, y, IMG_W, IMG_H, max_iter, cr, ci);
		int  color = round(juliaValue * 255);
		color = 255 - color;
		ptr[offset*4 + 0] = color;
		ptr[offset*4 + 1] = color;
		ptr[offset*4 + 2] = color;
		ptr[offset*4 + 3] = 255;
	}
}


int main(int argc, char *argv[]) {
	int device_count = -1; // number of devices

	device_count = ezc::Devices::instance().count();
	if (device_count == 0) {
		notify( "No CUDA devices found" );
	}
	
	eza::ArgumentParser parser(argv[0], "Julia Curves", argc, argv);

	std::vector<text> remaining_arguments; 
	 
	try {
		parser.add_integer("gpu", 'g', &device_id, 
			"Device used for display");
		parser.add_integer("block-size", 'b', &block_size, 
			"size of block in grid");
		parser.add_integer("image-width", 'w', &image_width, 
			"width of image to display");
		parser.add_integer("image-height", 'e', &image_height, 
			"height of image to display");
		parser.add_flag("view", 'v', &view_flag,
			"" );
		parser.add_integer("iterations", 'i', &iterations, 
			"number of iterations to determine if series converge");
		parser.add_real("constant-real", 'r', &constant_real,
			"real part of constant");
		parser.add_real("constant-imag", 's', &constant_imag,
			"imaginary part of constant");
		
		parser.parse(remaining_arguments);
		
	} catch(Exception& e) {
		parser.report_error(e);
		return EXIT_FAILURE;
	}
	
	ezc::Image *img = ezc::Image::get_instance();
	img->setup(image_height, image_width);

	eze::CPUTimer cpu_timer;
	ezc::Timer gpu_timer;

	cpu_timer.start();
	gpu_timer.start();

	GLubyte *gpu_pointer;
    cume_check( cudaMalloc( (void**)&gpu_pointer, img->size() ) );


	dim3 block(block_size, block_size);
    dim3 grid(image_width / block_size, image_height / block_size);


    kernel<<<grid,block>>>(gpu_pointer, image_width, image_height, iterations,
    	constant_real, constant_imag );
    cume_check_kernel();

    cume_check(cudaMemcpy( img->pointer(), gpu_pointer, img->size(),
                        cudaMemcpyDeviceToHost ) );

	cume_check(cudaMemcpy( img->pointer(), gpu_pointer, img->size(),
                        cudaMemcpyDeviceToHost ) );



	gpu_timer.stop();
	cpu_timer.stop();

	cout << "cpu time(ms)=" << cpu_timer << endl;
	cout << "gpu time(ms)=" << gpu_timer << endl;

	if (view_flag) {
		img->display("image", argc, argv);
	}

	cume_free( gpu_pointer );
		

	return EXIT_SUCCESS;
}
