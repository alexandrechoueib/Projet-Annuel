#ifndef GL_IMAGE_H
#define GL_IMAGE_H

#include "gl_helper.h"
#include <stdint.h>
#include <string>
#include <fstream>

#include "essential/import.h"

#ifdef LIBPNGPP
#include <png++/generator.hpp>
#include <png++/rgba_pixel.hpp>
#include <png++/writer.hpp>
#endif

using namespace eze;

typedef struct {
	byte red, green, blue, alpha;
} ImagePoint;

/**
 *
 */
class GLImage {
public:
	integer m_width, m_height;
	natural m_size;
	ImagePoint *m_pixels;
	
	GLImage(integer width, integer height);
	
	~GLImage() {
		delete [] m_pixels;
	}
	
	ImagePoint *pixels() {
		return m_pixels;
	}
	
	integer width() {
		return m_width;
	}
	
	integer height() {
		return m_height;
	}
	
	natural size() {
		return m_size;
	}
	
	void pixel(integer x, integer y, integer r, integer g, integer b) {
		int offset = y*m_width+x;
		m_pixels[offset].red= r;
		m_pixels[offset].green= g;
		m_pixels[offset].blue= b;
		m_pixels[offset].alpha= 255;
	}
	
	void pixel(integer x, integer y, integer r, integer g, integer b, integer a) {
		int offset = y*m_width+x;
		m_pixels[offset].red= r;
		m_pixels[offset].green= g;
		m_pixels[offset].blue= b;
		m_pixels[offset].alpha= a;
	}
	
	void draw() {
		glClearColor(0,0,0,1);
		glClear(GL_COLOR_BUFFER_BIT);
		glDrawPixels(m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, m_pixels);
		glutSwapBuffers();
	}

	void save(std::string file_name);

#ifdef LIBPNGPP
	void save_png(std::string file_name);
#endif
};

#endif

