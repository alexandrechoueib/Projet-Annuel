/*
 * test_dynamic_binary_tree_grapher.cpp
 *
 *  Created on: Aug 22, 2017
 *      Author: richer
 */



#include <gtest/gtest.h>
#include "essential/import.h"
#include "extensions/import.h"
#include "objects/import.h"
#include <vector>
#include "trees/dynamic_binary_tree.h"
#include "trees/dynamic_binary_tree_grapher.h"

using namespace std;
using namespace eze;
using namespace ezo;

typedef ez::trees::DynamicBinaryNode IDNode;
typedef ez::trees::DynamicBinaryTree IDTree;

TEST(DynamicBinaryTreeGrapher, degraph) {
	vector<IDNode *> v;

	for (integer i=5; i>0; --i) {
		v.push_back(new IDNode(i));
	}

	while (v.size() > 1) {
		IDNode *node = new IDNode();
		node->left(v[0]);
		node->right(v[1]);
		v.push_back(node);
		v.erase(v.begin());
		v.erase(v.begin());
	}

	IDTree t(v[0]);
	ostringstream oss;
	oss << t;

	text computed_result = oss.str();
	text expected_result = "((3,2):5,(1,(5,4):9):10):15";
	EXPECT_EQ(expected_result, computed_result);

	ez::trees::DynamicBinaryTreeGrapher grapher(t);
	vector<IDNode *> all_nodes;
	t.nodes(all_nodes, ez::trees::TreeConstants::ALL_NODES);
	for (natural i = 1; i < all_nodes.size(); ++i) {
		cerr << "degraph node " << *all_nodes[i] << endl;
		grapher.degraph(all_nodes[i]);
		cerr << "new tree is " << t << endl;
		t.validity();
		cerr << "undo degraph" << endl;
		grapher.undo_degraph();
		cerr << "back to tree " << t << endl;
		t.validity();
	}
}

TEST(DynamicBinaryTreeGrapher, degraph_regraph) {
	vector<IDNode *> v;

	for (integer i=5; i>0; --i) {
		v.push_back(new IDNode(i));
	}

	while (v.size() > 1) {
		IDNode *node = new IDNode();
		node->left(v[0]);
		node->right(v[1]);
		v.push_back(node);
		v.erase(v.begin());
		v.erase(v.begin());
	}

	IDTree t(v[0]);
	ostringstream oss;
	oss << t;

	text computed_result = oss.str();
	text expected_result = "((3,2):5,(1,(5,4):9):10):15";
	EXPECT_EQ(expected_result, computed_result);

	ez::trees::DynamicBinaryTreeGrapher grapher(t);
	vector<IDNode *> all_nodes;
	t.nodes(all_nodes, ez::trees::TreeConstants::ALL_NODES);


	for (natural i = 0; i < all_nodes.size(); ++i) {
		cerr << i << ": " << *all_nodes[i] << endl;
	}

	cerr << "degraph node " << *all_nodes[1] << endl;
	grapher.degraph(all_nodes[1]);
	cerr << "new tree is " << t << endl;
	t.validity();
	grapher.regraph_on(all_nodes[4]);
	cerr << "new tree " << t << endl;
	t.validity();

	cerr << "degraph node " << *all_nodes[1] << endl;
	grapher.degraph(all_nodes[1]);
	cerr << "new tree is " << t << endl;
	t.validity();
	grapher.regraph_on(all_nodes[7]);
	cerr << "new tree " << t << endl;
	t.validity();


}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}


