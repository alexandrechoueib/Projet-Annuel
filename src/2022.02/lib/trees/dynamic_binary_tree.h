/*
 * dynamic_binary_tree.h
 *
 *  Created on: Sep 6, 2017
 *      Author: richer
 */

#ifndef VERSION_2017_04_TREES_DYNAMIC_BINARY_TREE_H_
#define VERSION_2017_04_TREES_DYNAMIC_BINARY_TREE_H_

#include "../../version_2019.08/essential/types.h"
#include "../../version_2019.08/trees/dynamic_binary_node.h"
#include "../../version_2019.08/trees/generic_tree.h"

namespace ez {

namespace trees {

class DynamicBinaryTree : public GenericTree {
public:
	typedef DynamicBinaryTree self;
	typedef DynamicBinaryNode *Node;

	Node m_root;

	DynamicBinaryTree();
	DynamicBinaryTree(Node root);
	DynamicBinaryTree(const DynamicBinaryTree& obj);
	DynamicBinaryTree& operator=(const DynamicBinaryTree& obj);

	Node root(Node root);

	DynamicBinaryTree *clone();


	void internals(vector<Node>& v);
	void externals(vector<Node>& v);
	void all_nodes(vector<Node>& v);

	void degraph(Node src);
	void regraph(Node src, Node dst);

	void check();
	void print(std::ostream& out);

};

} // end of namespace trees

} // end of namespace ez



#endif /* VERSION_2017_04_TREES_DYNAMIC_BINARY_TREE_H_ */
