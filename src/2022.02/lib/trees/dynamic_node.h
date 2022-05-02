/*
 * dynamic_node.h
 *
 *  Created on: Sep 5, 2017
 *      Author: richer
 */

#ifndef VERSION_2017_04_TREES_DYNAMIC_NODE_H_
#define VERSION_2017_04_TREES_DYNAMIC_NODE_H_

#include <vector>

#include "../../version_2019.08/essential/import.h"
using namespace eze;

namespace ez {

namespace trees {

class DynamicNode {
public:
	typedef DynamicNode self;

	std::vector<self *> m_nodes;
	self *m_parent;

	DynamicNode();
	DynamicNode(const DynamicNode& obj);
	DynamicNode& operator=(const DynamicNode& obj);
	~DynamicNode();

	self *clone();

	void clean();

	bool is_root();
	bool is_leaf();

	self *parent();
	void parent(self *parent);

	void add(self *node);
	self *operator[](natural n);
	self *replace(natural position, self *node);
	self *remove(natural position);

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

#endif /* VERSION_2017_04_TREES_DYNAMIC_NODE_H_ */
