#include "gl_animation.h"

GLAnimation *GLAnimation::m_instance = nullptr;

void AnimDraw();
void AnimIdle();
void AnimExit();
void AnimMouse(int button, int state, int mouse_x, int mouse_y);
void AnimKey(unsigned char key, int x, int y);
	
GLAnimation::GLAnimation() {
	m_image = nullptr;
	m_anim_function = nullptr;
	m_exit_function = nullptr;
	m_frames = 0;
	m_time = 0.0;
	m_mouse_x = 0;
	m_mouse_y = 0;
}

void GLAnimation::run(int argc, char *argv[]) {
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
	glutInitWindowSize(m_image->width(), m_image->height());
	glutCreateWindow("Window");
	glutKeyboardFunc(AnimKey);
	glutDisplayFunc(AnimDraw);
	glutMouseFunc(AnimMouse);
	glutMainLoop();
}
	
void GLAnimation::run(int argc, char *argv[],
		GLAnimationFunction anim_function, GLAnimationFunction exit_function) {
	m_anim_function = anim_function;
	m_exit_function = exit_function;
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
	glutInitWindowSize(m_image->width(), m_image->height());
	glutCreateWindow("Window");
	glutKeyboardFunc(AnimKey);
	glutDisplayFunc(AnimDraw);
	glutMouseFunc(AnimMouse);
	glutIdleFunc(AnimIdle);
	glutMainLoop();
}

void AnimDraw() {
	GLAnimation& instance = GLAnimation::instance();
	if (instance.m_image != nullptr) {
		instance.m_image->draw();
	}
}

ez::cume::Timer gpu_timer;

void AnimIdle() {
	GLAnimation& instance = GLAnimation::instance();
	++instance.m_frames;
	gpu_timer.start();
	(*instance.m_anim_function)();
	gpu_timer.stop();
	instance.m_time += static_cast<double>(gpu_timer.elapsed());
	double avg = instance.frames() / instance.time();
	cout << "frame=" << instance.frames();
	cout << ",average.time=";
	cout << std::fixed  << avg << ",total.time=" << instance.time() << endl;
	glutPostRedisplay();
}

void AnimKey(unsigned char key, int x, int y) {
	switch (key) {
    case 27:
       AnimExit();
	}
}

void AnimMouse(int button, int state, int mouse_x, int mouse_y) {
	if (button == GLUT_LEFT_BUTTON) {
		GLAnimation::instance().mouse_x(mouse_x);
		GLAnimation::instance().mouse_y(mouse_y);
		cout << "mouse_x=" << mouse_x << endl;
		cout << "mouse_y=" << mouse_y << endl;
		
	}
}

void AnimExit() {
	GLAnimation& instance = GLAnimation::instance();
	cout << "animation.time=" << instance.time() << endl;
	cout << "animation.frames=" << instance.frames() << endl;
	if (instance.m_exit_function != nullptr) {
		(*instance.m_exit_function)();
	}
	exit(0);
}
