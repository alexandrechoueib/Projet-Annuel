/*
 * test_dynamic_tree.cpp
 *
 *  Created on: Aug 15, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "essential/import.h"
#include "extensions/import.h"
#include "objects/import.h"
#include <vector>
#include "trees/dynamic_tree.h"

using namespace std;
using namespace eze;
using namespace ezo;

typedef ez::trees::DynamicNode NodeType;
typedef ez::trees::DynamicTree TreeType;


TEST(TestDynamicTree, Constructor) {
	vector<NodeType *> v;

	for (integer i=5; i>0; --i) {
		v.push_back(new NodeType);
	}

	while (v.size() > 1) {
		NodeType *node = new NodeType;
		node->add(v[0]);
		node->add(v[1]);
		v.push_back(node);
		v.erase(v.begin());
		v.erase(v.begin());
	}

	cerr << *v[0] << endl;
	TreeType t(v[0]);
	ostringstream oss;
	oss << t;


	text computed_result = oss.str();
	text expected_result = "((,),(,(,)))";
	EXPECT_EQ(expected_result, computed_result);
}

TEST(TestDynamicTree, nodes) {
	vector<NodeType *> v;

	for (integer i=5; i>0; --i) {
		v.push_back(new NodeType);
	}

	while (v.size() > 1) {
		NodeType *node = new NodeType;
		node->add(v[0]);
		node->add(v[1]);
		v.push_back(node);
		v.erase(v.begin());
		v.erase(v.begin());
	}

	TreeType t(v[0]);
	std::vector<NodeType *> vexternals, vinternals, vallnodes;
	t.externals(vexternals);
	t.internals(vinternals);
	t.all_nodes(vallnodes);

	EXPECT_EQ(vexternals.size(), 5);
	EXPECT_EQ(vinternals.size(), 4);
	EXPECT_EQ(vallnodes.size(), 9);
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}



