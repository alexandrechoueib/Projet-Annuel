/*
 * test_dynamic_binary_tree.cpp
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

using namespace std;
using namespace eze;
using namespace ezo;

typedef ez::trees::DynamicBinaryNode<integer> IDNode;
typedef ez::trees::DynamicBinaryTree<integer> IDTree;


TEST(TestDynamicBinaryTree, Constructor) {
	vector<IDNode *> v;

	for (integer i=5; i>0; --i) {
		v.push_back(new IDNode(i));
	}

	while (v.size() > 1) {
		IDNode *node = new IDNode(v[0]->data() + v[1]->data());
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
}

TEST(TestDynamicBinaryTree, nodes) {
	vector<IDNode *> v;

	for (integer i=5; i>0; --i) {
		v.push_back(new IDNode(i));
	}

	while (v.size() > 1) {
		IDNode *node = new IDNode(v[0]->data() + v[1]->data());
		node->left(v[0]);
		node->right(v[1]);
		v.push_back(node);
		v.erase(v.begin());
		v.erase(v.begin());
	}

	IDTree t(v[0]);
	std::vector<IDNode *> vexternals, vinternals, vallnodes;
	t.nodes(vexternals, ez::trees::TreeConstants::EXTERNALS);
	t.nodes(vinternals, ez::trees::TreeConstants::INTERNALS);
	t.nodes(vallnodes, ez::trees::TreeConstants::ALL_NODES);

	EXPECT_EQ(vexternals.size(), 5);
	EXPECT_EQ(vinternals.size(), 4);
	EXPECT_EQ(vallnodes.size(), 9);
}

bool comparePtrToIDNode(IDNode *x, IDNode *y) {
	return x->m_data - y->m_data;
}

TEST(TestDynamicTree, clone) {
	ezo::Vector<IDNode *> v;

	for (integer i=5; i>0; --i) {
		v.put_last(new IDNode(i));
	}

	while (v.size() > 1) {
		IDNode *node = new IDNode(v[1]->data() + v[2]->data());
		node->left(v[1]);
		node->right(v[2]);
		v << node;
		v.remove_first();
		v.remove_first();
	}

	IDTree t1(v[1]);
	ostringstream oss;
	oss << t1;
	text computed_result = oss.str();

	IDTree *t2 = t1.clone();
	oss.str("");
	oss << *t2;
	text expected_result = oss.str();

	ezo::Vector<IDNode *> t1_allnodes, t2_allnodes, allnodes;
	t1.nodes(t1_allnodes, ez::trees::TreeConstants::ALL_NODES);
	t2->nodes(t2_allnodes, ez::trees::TreeConstants::ALL_NODES);
	allnodes.put_last(t1_allnodes);
	allnodes.put_last(t2_allnodes);
	EXPECT_EQ(allnodes.size(), 9+9);

	//ezx::print(cout, allnodes);
	natural ix = 0;
	for (auto e : allnodes) {
		cout << ix << " " << e << " => " << e->m_data;
		cout << endl;
		++ix;
	}

	EXPECT_TRUE(ezx::all_diff(allnodes));
	EXPECT_TRUE(ezx::all_match(t1_allnodes, t2_allnodes, comparePtrToIDNode));

}



int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}




