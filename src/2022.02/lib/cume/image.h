/*
 * image.h
 *
 *  Created on: Oct 15, 2016
 *      Author: richer
 */

#ifndef IMAGE_H_
#define IMAGE_H_

#include <cstdlib>
#include <iostream>
#include <cstring>
using namespace std;

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <GL/glext.h>

namespace ez {

namespace cume {

/**
 * This class is intended to be used to display a basic RVBA image
 * and handle interaction like left click to save image as ppm file
 */
class Image {
protected:
	// pointer to raw data as R, V, B, A
	GLubyte *pointer_;
	// width and height of image
	int width_, height_;
	// pointer to image instance
	static Image *image;

	/**
	 * default constructor
	 */
	Image();

	/**
	 * handle keyboard interaction like escape to stop
	 * image display (see the display method)
	 */
	static void keyboard(unsigned char key, int x, int y);

	/**
	 * handle mouse interaction like left click to save image
	 * as file
	 */
	static void mouse(int button, int state, int x, int y);

	/**
	 * display image on screen
	 */
	static void draw();

public:
	/**
	 * return instance of image
	 */
	static Image *get_instance();

	/**
	 * setup image information
	 * @param h height of the image
	 * @param w width of the image
	 */
	void setup(int h, int w);

	/**
	 * destructor
	 */
	~Image();

	/**
	 * return width of the image
	 */
	int width() {
		return width_;
	}

	/**
	 * return height of the image
	 */
	int height() {
		return height_;
	}

	/**
	 * return pointer to the image data in global memory
	 */
	GLubyte *pointer() {
		return pointer_;
	}

	/**
	 * sets pointer to the image data
	 */
	void pointer(GLubyte *p) {
		pointer_ = p;
	}

	int size() {
		return width_ * height_ * 4;
	}

	/**
	 * display image as window on screen and wait for user
	 * interaction
	 * @param title message of the window that displays the image
	 * @param argc number of command line parameters
	 * @param argv command line parameters
	 */
	void display(string title, int argc, char *argv[]);
};

} // end namespace cume

} // end namespace ez


#endif /* IMAGE_H_ */
