/*
 * dynamic_node.cpp
 *
 *  Created on: Sep 6, 2017
 *      Author: richer
 */

#include "../../version_2019.08/trees/dynamic_node.h"

using namespace ez::trees;

DynamicNode::DynamicNode() : m_parent(nullptr) {

}

DynamicNode::DynamicNode(const DynamicNode& obj) : m_parent(nullptr) {
	for (auto e : obj.m_nodes) {
		m_nodes.push_back(e->clone());
		e->parent(this);
	}
}

DynamicNode& DynamicNode::operator=(const DynamicNode& obj) {
	if (&obj != this) {
		notify("not implemented");
	}
	return *this;
}

DynamicNode::~DynamicNode() {

}

DynamicNode *DynamicNode::clone() {
	DynamicNode *new_node = new DynamicNode;
	for (auto e : m_nodes) {
		new_node->m_nodes.push_back(e->clone());
		e->parent(new_node);
	}
	return new_node;
}

void DynamicNode::clean() {
	for (auto e : m_nodes) {
		e->clean();
		delete e;
	}
}

DynamicNode *DynamicNode::parent() {
	return m_parent;
}

void DynamicNode::parent(DynamicNode *parent) {
	m_parent = parent;
}

bool DynamicNode::is_root() {
	return m_parent == nullptr;
}

bool DynamicNode::is_leaf() {
	return m_nodes.size() == 0;
}

void DynamicNode::add(DynamicNode *node) {
	ensure(node != nullptr);
	m_nodes.push_back(node);
	node->parent(this);
}

DynamicNode *DynamicNode::operator[](natural position) {
	ensure((1 <= position) && (position <= m_nodes.size()));
	return m_nodes[position - 1];
}

DynamicNode *DynamicNode::replace(natural position, self *node) {
	ensure(node != nullptr);
	ensure((1 <= position) && (position <= m_nodes.size()));
	--position;
	DynamicNode *result = m_nodes[position];
	m_nodes[position] = node;
	node->parent(this);
	return result;
}

DynamicNode *DynamicNode::remove(natural position) {
	ensure((1 <= position) && (position <= m_nodes.size()));
	--position;
	DynamicNode *result = m_nodes[position];
	m_nodes.erase(m_nodes.begin() + position);
	return result;
}

void DynamicNode::print(std::ostream& out) {
	if (!is_leaf()) {
		out << "(";
		m_nodes[0]->print(out);
		natural i = 1;
		while (i < m_nodes.size()) {
			out << ",";
			m_nodes[i]->print(out);
			++i;
		}
		out << ")";
	} else {
		//out << "_";
	}
}

void DynamicNode::check() {
	if (is_leaf()) return ;
	for (auto e : m_nodes) {
		if (e->parent() != this) {
			notify("node " << e << " does not have " << this << " as parent" );
		}
		e->check();
	}
}


void DynamicNode::internals(vector<DynamicNode *>& v) {
	if (!is_leaf()) {
		v.push_back(this);
		for (auto e : m_nodes) {
			e->externals(v);
		}
	}

}

void DynamicNode::externals(vector<DynamicNode *>& v) {
	if (is_leaf()) {
		v.push_back(this);
	} else {
		for (auto e : m_nodes) {
			e->internals(v);
		}
	}
}

void DynamicNode::all_nodes(vector<DynamicNode *>& v) {
	if (is_leaf()) {
		v.push_back(this);
	} else {
		v.push_back(this);
		for (auto e : m_nodes) {
			e->all_nodes(v);
		}
	}
}

