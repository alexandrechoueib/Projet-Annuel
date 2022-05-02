// ==================================================================
// CUME V2017
// Copyright 2015-2017 Jean-Michel RICHER
// License: BSD License
// Please see the README file for documentation
// ==================================================================

#include <cuda.h>
#include "common/cpu_anim.h"
#include "common/cpu_bitmap.h"
#include "cume.h"
#include <getopt.h>

using namespace cume;

// ==================================================================
// global variables
// ==================================================================
// width of the image in pixels
int IMAGE_WIDTH  = 256;
// height of the image in pixels
int IMAGE_HEIGHT = 256;
// display image, set to false to perform only calculations
bool display = true;

// ==================================================================
// program command line arguments
// ==================================================================
static struct option long_options[] = {
  {"width"  , required_argument, 0, 'w'},
  {"height" , required_argument, 0, 'h'},
  {"no-show", no_argument,       0, 'n'},
  {0,0,0,0}
 
};

struct aComplex {
	float r;
	float i;
	
	__device__ aComplex( float a, float b ) : r(a), i(b) {}
	__device__ float magnitude2( void ) { return r * r + i * i; }
	__device__ aComplex operator*(const aComplex& a) {
		return aComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	__device__ aComplex operator+(const aComplex& a) {
		return aComplex(r+a.r, i+a.i);
	}
};

/**
 * determine if given point is in Julia Set
 *
 */
 
__device__ int julia( int x, int y, int IMG_W, int IMG_H ) {
	const float scale = 0.1;
	float jx = scale * (float)(IMG_W/2 - x)/(IMG_W/2);
	float jy = scale * (float)(IMG_H/2 - y)/(IMG_H/2);

	aComplex c(-0.8, 0.156);	
	
	aComplex z(jx, jy);
	int i = 0; 
	for (i=0; i<200; i++) {
		z = z * z + c;
		if (z.magnitude2() > 10000.0) return 0;
	}
	return 1;
}

__global__ void kernel( unsigned char *ptr, int IMG_W, int IMG_H) {
	// map from threadIdx/BlockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * gridDim.x * blockDim.x;
	
	// now calculate the value at that position
	int juliaValue = julia( x, y, IMG_W, IMG_H );
	ptr[offset*4 + 0] = 255 * juliaValue;
	ptr[offset*4 + 1] = 0;
	ptr[offset*4 + 2] = 0;
	ptr[offset*4 + 3] = 255; 
}

int main(int argc, char *argv[]) {
	int option_index;
	while (true) {
		option_index = 0;
        int c = getopt_long(argc, argv, "w:h:n", long_options, &option_index);
        if (c == -1) break;
 
		switch(c) {
		case 'w':
        	IMAGE_WIDTH = atoi(optarg);
        	if (IMAGE_WIDTH < 128) IMAGE_WIDTH = 128;
        	break;
      	case 'h':
        	IMAGE_HEIGHT = atof(optarg);
        	if (IMAGE_HEIGHT < 128) IMAGE_HEIGHT = 128;
        	break;
      	case 'n':
        	display = false;
        	break;
      
    	}
  	}
  	
  	cout << "width=" << IMAGE_WIDTH << endl;
  	cout << "height=" << IMAGE_HEIGHT << endl;
	
	CPUBitmap bitmap(IMAGE_WIDTH, IMAGE_HEIGHT);

	unsigned char *dev_bitmap;

	cume_new_array(dev_bitmap, unsigned char, bitmap.image_size() );

	Kernel k(IMAGE_WIDTH * IMAGE_HEIGHT);
	k.configure(GRID_XY, BLOCK_XY, IMAGE_WIDTH/16, IMAGE_HEIGHT/16, 16, 16);
	k.set_timer_needed();
	
	kernel_call_no_resource(kernel, k, dev_bitmap, IMAGE_WIDTH, IMAGE_HEIGHT );
	
	cume_pull( bitmap.get_ptr(), dev_bitmap, unsigned char, bitmap.image_size() );
			
	if (display) {
		bitmap.display_and_exit();
	}	
	
	cume_free( dev_bitmap );
	
	return 0;
}

