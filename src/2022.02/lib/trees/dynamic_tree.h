/*
 * dynamic_tree.h
 *
 *  Created on: Sep 5, 2017
 *      Author: richer
 */

#ifndef VERSION_2017_04_TREES_DYNAMIC_TREE_H_
#define VERSION_2017_04_TREES_DYNAMIC_TREE_H_

#include "../../version_2019.08/essential/types.h"
#include "../../version_2019.08/trees/dynamic_node.h"
#include "../../version_2019.08/trees/generic_tree.h"

namespace ez {

namespace trees {

class DynamicTree : public GenericTree {
public:
	typedef DynamicTree self;
	typedef DynamicNode *Node;

	Node m_root;

	DynamicTree();
	DynamicTree(Node root);
	DynamicTree(const DynamicTree& obj);
	DynamicTree& operator=(const DynamicTree& obj);

	Node root(Node root);

	DynamicTree *clone();

	void print(std::ostream& out);

	void check();

	void internals(vector<Node>& v);
	void externals(vector<Node>& v);
	void all_nodes(vector<Node>& v);

};

} // end of namespace trees

} // end of namespace ez

#endif /* VERSION_2017_04_TREES_DYNAMIC_TREE_H_ */
