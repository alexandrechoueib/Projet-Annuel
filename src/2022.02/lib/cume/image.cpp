#include "image.h"

using namespace ez::cume;

void free_resource() {
	Image *image = Image::get_instance();
	if (image != nullptr) {
		if (image->pointer() != nullptr) {
			delete image->pointer();
			image->pointer(nullptr);
		}
	}
}

void generate_ppm(int width, int height, GLubyte *p) {
	cout << "try to save image on image.ppm file" << endl;
	const char *file_name = "image.ppm";
	FILE *f = fopen(file_name, "w");
	if (f == nullptr) {
		cout << "error: could not open file for writing !" << endl;
		return ;
	}
	fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
	int offset = 0;
	for (int y=0; y<height; ++y) {
		for (int x=0; x<width; ++x) {
			fprintf(f, "%3d %3d %3d\n", p[offset], p[offset+1], p[offset+2]);
			offset += 4;
		}
	}
	fclose(f);
	cout << "done !" << endl;
}

Image::Image() : pointer_(nullptr), width_(0), height_(0) {
}

Image *Image::get_instance() {
	if (image == nullptr) {
		image = new Image();
	}
	return image;
}

void Image::setup(int h, int w) {
	width_ = w;
	height_ = h;
	pointer_ = new GLubyte[ w * h * 4];
	memset(pointer_, 0, w * h * 4);

}

Image::~Image() {
	delete [] pointer_;
}

void Image::keyboard(unsigned char key, int x, int y) {
	switch(key) {
		case 27: /* ESC */
		case 81: /* Q */
		case 113: /* q */
			exit(EXIT_SUCCESS);
			break;
	}
}

void Image::mouse(int button, int state, int x, int y) {
	if (state == GLUT_DOWN) {
		// create image
		Image *img = Image::get_instance();
		generate_ppm(img->width(), img->height(), img->pointer());
	}
}

void Image::draw() {
	Image *img = Image::get_instance();
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(img->width(), img->height(), GL_RGBA, GL_UNSIGNED_BYTE, img->pointer());
	// use glutSwapBuffers(); if glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glFlush();
}


void Image::display(string title, int argc, char *argv[]) {
	glutInit(&argc, argv);
	glutInitWindowSize(width_, height_);
	glutInitWindowPosition(50,50);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
	glutCreateWindow(title.c_str());
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutDisplayFunc(draw);
	atexit(free_resource);
	cout << "use mouse click to save image as file and escape to exit..." << endl;
	glutMainLoop();
}

Image *Image::image = nullptr;

