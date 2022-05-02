/*
 * node.h
 *
 *  Created on: Apr 11, 2017
 *      Author: Jean-Michel Richer
 */

#ifndef OBJECTS_NODE_H_
#define OBJECTS_NODE_H_

#include "objects/object.h"

namespace ez {

namespace objects {

/**
 * implementation of a Node used by a Tree. The Node
 * can have 0 to MaxNodes sibling.
 */
template<class T, natural MaxNodes>
class Node : public Object {
public:
	typedef Node<T, MaxNodes> self;

	// create C-array of nodes where node at position 0
	// is the parent
	self *c_nodes[MaxNodes + 1];
	// number of nodes
	natural n_size;
	// data stored by the node
	T t_data;

	enum {
		PARENT = 0, LEFT = 1, RIGHT = 2
	};


	/**
	 * constructor with given data
	 */
	Node(T x) : Object(), t_data(x) {
		for (natural i = 0; i <= MaxNodes; ++i) {
			c_nodes[i] = nullptr;
		}
		n_size = 0;
	}

	/**
	 * copy constructor
	 */
	Node(const self& obj) : Object(), t_data(obj.t_data) {
		n_size = obj.n_size;
		natural i;
		for (i = 0; i <= n_size; ++i) {
			c_nodes[i] = obj.c_nodes[i];
		}
		for (; i <= MaxNodes; ++i) {
			c_nodes[i] = nullptr;
		}
	}

	/**
	 * assignment operator
	 */
	self& operator=(const self& obj) {
		if (&obj != this) {
			t_data = obj.t_data;
			n_size = obj.n_size;

			natural i;
			for (i = 0; i <= n_size; ++i) {
				c_nodes[i] = obj.c_nodes[i];
			}
			for (; i <= MaxNodes; ++i) {
				c_nodes[i] = nullptr;
			}
		}
		return *this;
	}

	~Node() {

	}

	/**
	 * return true if node is leaf, i.e. has no descendant
	 */
	bool is_leaf() {
		return n_size == 0;
	}

	/**
	 * return true if node is root, i.e. has no parent
	 */
	bool is_root() {
		return c_nodes[PARENT] == nullptr;
	}

	/**
	 * add sibling
	 */
	void add(self *obj) {
		ensure(n_size < MaxNodes);
		c_nodes[1 + n_size++] = obj;
		if (obj != nullptr) {
			obj->c_nodes[PARENT] = this;
		}
	}

	/**
	 * remove node at given position
	 */
	self *remove(integer pos) {
		ensure((pos != PARENT) && (pos <= MaxNodes));
		self *tmp = c_nodes[pos];
		natural i = pos+1;
		while (i <= n_size) {
			c_nodes[pos++] = c_nodes[i++];
		}
		c_nodes[pos] = nullptr;
		return tmp;
	}

	/**
	 * return depth of node
	 */
	natural depth() {
		if (n_size == 0) return 1;
		natural d = 0;
		for (natural i = 1; i <= n_size; ++i) {
			d = std::max(d, c_nodes[i]->depth());
		}
		return d + 1;
	}

	/**
	 * clone tree that starts from this node
	 */
	self *clone() {
		self *new_node = new self(t_data);
		new_node->c_nodes[PARENT] = nullptr;
		new_node->n_size = n_size;
		natural i;
		for (i = 1; i <= n_size; ++i) {
			new_node->c_nodes[i] = c_nodes[i]->clone();
		}
		return new_node;
	}

	/**
	 * print subtree
	 */
	void print(std::ostream& out) {
		if (is_leaf()) {
			out << t_data;
		} else {
			out << "(";
			if (n_size > 0) {
				c_nodes[1]->print(out);
				for (natural i = 2; i <= n_size; ++i) {
					out << ",";
					c_nodes[i]->print(out);
				}
			}
			out << ")";
		}
	}

	void output(std::ostream& out) {
	}

	void input(std::istream& out) {
	}

	integer compare(const Object& y) {
		self *y_ = dynamic_cast<self *>(&const_cast<Object&>(y));
		return (t_data == y_->t_data);
	}

	/**
	 * remove all descendants and delete them
	 */
	void remove_all() {
		for (natural i = 1; i<= n_size; ++i) {
			c_nodes[i]->remove_all();
			delete c_nodes[i];
		}
		n_size = 0;
	}

	/**
	 * set number of sibling to 0 but don't delete sibling
	 */
	void empty() {
		n_size = 0;
	}

	bool is_empty() {
		return n_size == 0;
	}

	bool belongs_to(Node<T,MaxNodes> *n) {
		if (n == this) return true;
		for (natural i = 1; i<= n_size; ++i) {
			if (c_nodes[i]->belongs_to(n)) return true;
		}
		return false;
	}

	void find_nodes(std::vector<Node<T,MaxNodes> *>& vnodes) {
		vnodes.push_back(this);
		for (natural i = 1; i <= n_size; ++i) {
			c_nodes[i]->find_nodes(vnodes);
		}
	}

	void find_externals(std::vector<Node<T,MaxNodes> *>& vnodes) {
		if (this->is_leaf()) {
			vnodes.push_back(this);
		} else {
			for (natural i = 1; i <= n_size; ++i) {
				c_nodes[i]->find_nodes(vnodes);
			}
		}
	}

	void find_internals(std::vector<Node<T,MaxNodes> *>& vnodes) {
		if (!this->is_leaf()) {
			for (natural i = 1; i <= n_size; ++i) {
				c_nodes[i]->find_nodes(vnodes);
			}
		}
	}

	// =======================
	// Operations for
	// binary nodes
	// =======================

	Node<T,MaxNodes> *sibling(Node<T,MaxNodes> *n) {
		assume(c_nodes->n_size == 2);
		assume((c_nodes[LEFT] == n) || (c_nodes[RIGHT] == n));

		if (c_nodes[LEFT] == n) {
			return c_nodes[RIGHT];
		} else {
			return c_nodes[LEFT];
		}
	}


};

} // end of namespace objects

} // end of namespace ez


#endif /* OBJECTS_NODE_H_ */
