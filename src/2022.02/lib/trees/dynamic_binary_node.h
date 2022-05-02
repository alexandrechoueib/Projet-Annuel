/*
 * dynamic_binary_node.h
 *
 *  Created on: Sep 6, 2017
 *      Author: richer
 */

#ifndef VERSION_2017_04_TREES_DYNAMIC_BINARY_NODE_H_
#define VERSION_2017_04_TREES_DYNAMIC_BINARY_NODE_H_

#include <vector>

#include "../../version_2019.08/essential/import.h"
using namespace eze;

namespace ez {

namespace trees {

class DynamicBinaryNode {
public:
	typedef DynamicBinaryNode self;

	enum { LEFT = 0, RIGHT = 1, PARENT = 2, ALIGNMENT = 4 };

	self *m_nodes[ALIGNMENT];

	DynamicBinaryNode();
	DynamicBinaryNode(const DynamicNode& obj);
	DynamicBinaryNode& operator=(const DynamicBinaryNode& obj);
	~DynamicBinaryNode();

	self *clone();

	void clean();

	bool is_root();
	bool is_leaf();

	self *parent();
	void parent(self *parent);

	self *left();
	void left(self *node);

	self *right();
	void right(self *node);

	void print(std::ostream& out);

	friend std::ostream& operator<<(std::ostream& out, self& obj) {
		obj.print(out);
		return out;
	}

	void check();

	void internals(vector<self *>& v);
	void externals(vector<self *>& v);
	void all_nodes(vector<self *>& v);

};

} // end of namespace trees

} // end of namespace ez




#endif /* VERSION_2017_04_TREES_DYNAMIC_BINARY_NODE_H_ */
