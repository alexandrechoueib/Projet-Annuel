/*
 * test_geometry3d.cpp
 *
 *  Created on: Aug 9, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "objects/vector.h"
#include "maths/import.h"
#include <vector>
#include "essential/import.h"

using namespace std;
using namespace ez;


namespace ezo = ez::objects;


TEST(TestGeometry3D, rotation_x) {
	ezm::Point3D p1(1.0, 0.0, 1.0);
	ezm::Point3D p2;
	ezm::Geometry3D::GMatrix rotx, roty, rotz;

	ezm::Geometry3D::matrix_rotate_x(ezm::Constants::ANGLE_DEGREE_90, rotx);
	ezm::Geometry3D::matrix_rotate_y(ezm::Constants::ANGLE_DEGREE_90, roty);
	ezm::Geometry3D::matrix_rotate_z(ezm::Constants::ANGLE_DEGREE_90, rotz);

	cout << "rotx=" << endl << rotx << endl;

	ezm::Geometry3D::product(p2, rotx, p1);
	cout << "p2=" << p2 << endl;

	cout << "roty=" << endl << roty << endl;
	ezm::Geometry3D::product(p2, roty, p1);
	cout << "p2=" << p2 << endl;

	cout << "rotz=" << endl << rotz << endl;
	ezm::Geometry3D::product(p2, rotz, p1);
	cout << "p2=" << p2 << endl;

}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}



