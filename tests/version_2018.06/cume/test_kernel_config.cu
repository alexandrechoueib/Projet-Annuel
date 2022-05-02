#include <gtest/gtest.h>
#include "cume/import.h"

using namespace ezc;

/** 
 * kernel that computes the sum of vectors c = a + b
 */
__global__ void kernel_sum_1_x(ezc::Kernel::Resource *r, int *a, int *b, int *c, int size) {
	//int tid = gtid_x_x();
	int tid = r->get_global_tid();
	
	if (tid < size) {
		c[tid] = a[tid] + b[tid];
	}
}

/**
 * kernel that computes the sum of vectors c = a + b
 */
__global__ void kernel_sum_x_1(ezc::Kernel::Resource *r, int *a, int *b, int *c, int size) {
	//int tid = gtid_x_1();
	int tid = r->get_global_tid();
	
	if (tid < size) {
		c[tid] = a[tid] + b[tid];
	} 
}

/**
 * kernel that computes the sum of vectors c = a + b
 */
__global__ void kernel_sum_1_xy(ezc::Kernel::Resource *r, int *a, int *b, int *c, int size) {
	//int tid = gtid_1_xy();
	int tid = r->get_global_tid();
	
	if (tid < size) {
		c[tid] = a[tid] + b[tid];
	}
}

/**
 * kernel that computes the sum of vectors c = a + b
 */
__global__ void kernel_sum_xy_1(ezc::Kernel::Resource *r, int *a, int *b, int *c, int size) {
	//int tid = gtid_xy_1();
	int tid = r->get_global_tid();
	
	if (tid < size) {
		c[tid] = a[tid] + b[tid];
	}
}

/**
 * kernel that computes the sum of vectors c = a + b
 */
__global__ void kernel_sum_x_x(ezc::Kernel::Resource *r, int *a, int *b, int *c, int size) {
	//int tid = gtid_x_x();
	int tid = r->get_global_tid();
	
	if (tid < size) {
		c[tid] = a[tid] + b[tid];
	}
}

/**
 * kernel that computes the sum of vectors c = a + b
 */
__global__ void kernel_sum_xy_x(ezc::Kernel::Resource *r, int *a, int *b, int *c, int size) {
	//int tid = gtid_xy_x();
	int tid = r->get_global_tid();
	
	if (tid < size) {
		c[tid] = a[tid] + b[tid];
	}
} 

/**
 * kernel that computes the sum of vectors c = a + b
 */
__global__ void kernel_sum_x_xy(ezc::Kernel::Resource *r, int *a, int *b, int *c, int size) {
	//int tid = gtid_x_xy();
	int tid = r->get_global_tid();
	
	if (tid < size) {
		c[tid] = a[tid] + b[tid];
	}
}

/**
 * kernel that computes the sum of vectors c = a + b
 */
__global__ void kernel_sum_xy_xy(ezc::Kernel::Resource *r, int *a, int *b, int *c, int size) {
	//int tid = gtid_xy_xy();
	int tid = r->get_global_tid();
	
	if (tid < size) {
		c[tid] = a[tid] + b[tid];
	}
}


/**
 * kernel that computes the sum of vectors c = a + b
 */
__global__ void kernel_sum_xyz_1(ezc::Kernel::Resource *r, int *a, int *b, int *c, int size) {
	//int tid = gtid_xyz_1();
	int tid = r->get_global_tid();
	
	if (tid < size) {
		c[tid] = a[tid] + b[tid];
	}
}


/**
 *
 */
template<class T>
bool check(Array<T>& c) {
	int sum = accumulate(c.begin(), c.end(), 0);
	int expected_sum = 4*c.size()*(c.size() + 1)/2;
	//cout << "sum = " << sum << ", exp = " << expected_sum << ", size = " << c.size() << endl;
	cout << "sum = " << sum << endl;
	
	
	 
	return (sum == expected_sum);
}


#define init(str, size) \
	const int SIZE = size; \
	ez::cume::Array<int> a(SIZE), b(SIZE), c(SIZE); \
	cout << str; \
	std::iota(a.begin(), a.end(), 1); \
	std::iota(b.begin(), b.end(), 1); \
	std::transform(b.begin(), b.end(), b.begin(), std::bind1st(std::multiplies<int>(), 3)); \
	a.push(); \
	b.push(); \
	Kernel k(SIZE);

/**
 * check cume thread index formula for 1_x which corresponds
 * to grid(1,1,1) block(x,1,1)
 *
 */
TEST(TestCUDA, check_kernel_1_x) {
	init("test kernel_sum_1_x: ", 512);
	
	k.configure(GRID_1, BLOCK_X, SIZE);
	kernel_call(kernel_sum_1_x, k, &a, &b, &c, a.size());
	c.pull();
	EXPECT_TRUE(check(c));
}

/**
 * check cume thread index formula for x_1 which corresponds
 * to grid(x,1,1) block(1,1,1)
 *
 */
TEST(TestCUDA, check_kernel_x_1) {
	init("test kernel_sum_x_1: ", 512);

	k.configure(GRID_X, BLOCK_1, SIZE);
	kernel_call(kernel_sum_x_1, k, &a, &b, &c, a.size());
	
	c.pull();
	EXPECT_TRUE(check(c));
}

/**
 * check cume thread index formula for 1_xy which corresponds
 * to grid(1,1,1) block(x,y,1)
 *
 */
TEST(TestCUDA, check_kernel_1_xy) {
	init("test kernel_sum_1_xy: ", 512);

	k.configure(GRID_1, BLOCK_XY, 1, SIZE);
	kernel_call(kernel_sum_1_xy, k, &a, &b, &c, a.size());
	
	c.pull();
	EXPECT_TRUE(check(c));
}

/**
 * check cume thread index formula for xy_1 which corresponds
 * to grid(x,y,1) block(1,1,1)
 *
 */
TEST(TestCUDA, check_kernel_xy_1) {
	init("test kernel_sum_xy_1: ", 512);

	k.configure(GRID_XY, BLOCK_1, 128, 4);
	kernel_call(kernel_sum_xy_1, k, &a, &b, &c, a.size());
	
	c.pull();
	EXPECT_TRUE(check(c));
}

/**
 * check cume thread index formula for xy_x which corresponds
 * to grid(x,y,1) block(x',1,1)
 *
 */
TEST(TestCUDA, check_kernel_xy_x) {
	init("test kernel_sum_xy_x: ", 16384);

	k.configure(GRID_XY, BLOCK_X, 16, 32, 32);
	kernel_call(kernel_sum_xy_x, k, &a, &b, &c, a.size());
	
	c.pull();
	EXPECT_TRUE(check(c));
}

/**
 * check cume thread index formula for x_x which corresponds
 * to grid(x,1,1) block(x',1,1)
 *
 */
TEST(TestCUDA, check_kernel_x_x) {
	init("test kernel_sum_x_x: ", 2048*8);

	k.configure(GRID_X, BLOCK_X, SIZE/512, 512);
	kernel_call(kernel_sum_x_x, k, &a, &b, &c, a.size());
	
	c.pull();
	EXPECT_TRUE(check(c));
}

/**
 * check cume thread index formula for xy_x which corresponds
 * to grid(x,1,1) block(x',y',1)
 *
 */ 
TEST(TestCUDA, check_kernel_x_xy) {
	init("test kernel_sum_x_xy: ", 8192);

	k.configure(GRID_X, BLOCK_XY, 8, 32, 32);
	kernel_call(kernel_sum_x_xy, k, &a, &b, &c, a.size());
	
	c.pull();
	EXPECT_TRUE(check(c));
}

/**
 * check cume thread index formula for xy_xy which corresponds
 * to grid(x,y,1) block(x',y',1)
 *
 */ 
TEST(TestCUDA, check_kernel_xy_xy) {
	init("test kernel_sum_xy_xy: ", 8192);

	k.configure(GRID_XY, BLOCK_XY, 8, 2, 16, 32);
	kernel_call(kernel_sum_xy_xy, k, &a, &b, &c, a.size());
	
	c.pull();
	EXPECT_TRUE(check(c));
}

/**
 * check cume thread index formula for xyz_1 which corresponds
 * to grid(x,y,z) block(1,1,1)
 *
 */ 
TEST(TestCUDA, check_kernel_xyz_1) {
	init("test kernel_sum_xyz_1: ", 8192);

	k.configure(GRID_XYZ, BLOCK_1, 8, 512, 2);
	kernel_call(kernel_sum_xyz_1, k, &a, &b, &c, a.size());

	c.pull();
	EXPECT_TRUE(check(c));
}

/**
 * main function
 */
int main(int argc, char *argv[]) {
	
	cout << "Test of kernel global thread index formulae" << endl;
		
	cout << Devices::get_instance() << endl;
	
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}