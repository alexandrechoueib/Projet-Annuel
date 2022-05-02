/*
 * test_node.cpp
 *
 *  Created on: Apr 11, 2017
 *      Author: richer
 */


#include <gtest/gtest.h>
#include "objects/integer.h"
#include "objects/node.h"
#include <vector>

using namespace std;

namespace eze = ez::essential;
namespace ezo = ez::objects;

typedef ezo::Node<eze::integer, 2> IntegerBinaryNode;

TEST(TestNode, create1) {
	IntegerBinaryNode *a, *b, *c;

	a = new IntegerBinaryNode(1);
	b = new IntegerBinaryNode(2);
	c = new IntegerBinaryNode(3);

	c->add(a);
	c->add(b);

	cerr << *c << endl;
}

TEST(TestNode, create2) {
	IntegerBinaryNode *a, *b, *c, *d, *e;

	a = new IntegerBinaryNode(1);
	b = new IntegerBinaryNode(2);
	c = new IntegerBinaryNode(3);
	d = new IntegerBinaryNode(4);
	e = new IntegerBinaryNode(7);

	c->add(a);
	c->add(b);
	e->add(c);
	e->add(d);

	vector<IntegerBinaryNode *> nodes;
	nodes.push_back(a);
	nodes.push_back(b);
	nodes.push_back(c);
	nodes.push_back(d);
	nodes.push_back(e);

	cerr << *e << endl;
	for (eze::natural i = 0; i<nodes.size(); ++i) {
		cerr << "depth(" << *nodes[i] << ")=" << nodes[i]->depth() << endl;
	}

	IntegerBinaryNode *f = e->clone();

	cout << *f << endl;
}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}


