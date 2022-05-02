#ifndef GL_ANIMATION_H
#define GL_ANIMATION_H

#include "gl_image.h"
#include "timer.h"

typedef void (*GLAnimationFunction)(void);

class GLAnimation {
private:
	static GLAnimation *m_instance;
		
	GLAnimation();
	
	
public:
	GLImage *m_image;
	GLAnimationFunction m_anim_function, m_exit_function;
	uint32_t m_frames;
	double m_time;
	int m_mouse_x, m_mouse_y;
		
	static GLAnimation& instance() {
		if (m_instance == nullptr) {
			m_instance = new GLAnimation;
		}
		return *m_instance;
	}
	
	GLImage * image() {
		return m_image;
	}
	
	void image(GLImage *image) {
		m_image = image;
	}
	
	uint32_t frames() {
		return m_frames;
	}
	
	void mouse_x(int x) {
		m_mouse_x = x;
	}
	
	void mouse_y(int y) {
		m_mouse_y = y;
	}
	
	int mouse_x() {
		return m_mouse_x;
	}
	
	int mouse_y() {
		return m_mouse_y;
	}
	
	/**
	 * return time in milli seconds
	 */
	double time() {
		return m_time;
	}
	
	void run(int argc, char *argv[]);
	
	void run(int argc, char *argv[], GLAnimationFunction anim_function,
			GLAnimationFunction exit_function = nullptr);
	
};

#endif
