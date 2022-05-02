/*
 * dynamic_binary_node.cpp
 *
 *  Created on: Sep 6, 2017
 *      Author: richer
 */

#include "../../version_2019.08/trees/dynamic_binary_node.h"

using namespace ez::trees;

DynamicBinaryNode::DynamicBinaryNode() {
	m_nodes[LEFT] = m_nodes[RIGHT] = m_nodes[PARENT] = nullptr;
}

DynamicBinaryNode::DynamicBinaryNode(const DynamicBinaryNode& obj) {
	m_nodes[LEFT] = m_nodes[RIGHT] = m_nodes[PARENT] = nullptr;
	if (obj.m_nodes[LEFT] != nullptr) {
		m_nodes[LEFT] = obj.m_nodes[LEFT]->clone();
		m_nodes[LEFT]->parent(this);
	}
	if (obj.m_nodes[RIGHT] != nullptr) {
		m_nodes[RIGHT] = obj.m_nodes[RIGHT]->clone();
		m_nodes[RIGHT]->parent(this);
	}
}

DynamicBinaryNode& DynamicBinaryNode::operator=(const DynamicBinaryNode& obj) {
	if (&obj != this) {
		notify("not implemented");
	}
	return *this;
}

DynamicBinaryNode::~DynamicBinaryNode() {

}

DynamicBinaryNode *DynamicBinaryNode::clone() {
	DynamicBinaryNode *new_node = new DynamicBinaryNode;

	return new_node;
}

void DynamicBinaryNode::clean() {

}

DynamicBinaryNode *DynamicBinaryNode::parent() {
	return m_nodes[PARENT];
}

void DynamicBinaryNode::parent(DynamicBinaryNode *parent) {
	m_nodes[PARENT] = parent;
}

DynamicBinaryNode *DynamicBinaryNode::left() {
	return m_nodes[LEFT];
}

void DynamicBinaryNode::left(DynamicBinaryNode *parent) {
	m_nodes[LEFT] = parent;
}

DynamicBinaryNode *DynamicBinaryNode::right() {
	return m_nodes[RIGHT];
}

void DynamicBinaryNode::right(DynamicBinaryNode *parent) {
	m_nodes[RIGHT] = parent;
}



bool DynamicBinaryNode::is_root() {
	return m_nodes[PARENT] == nullptr;
}

bool DynamicBinaryNode::is_leaf() {
	return ((m_nodes[LEFT] == nullptr) && (m_nodes[RIGHT] == nullptr));
}


void DynamicBinaryNode::print(std::ostream& out) {
	if (!is_leaf()) {
		out << "(";
		m_nodes[LEFT]->print(out);
		out << ",";
		m_nodes[RIGHT]->print(out);
		out << ")";
	} else {
		out << "_";
	}
}

void DynamicBinaryNode::check() {
	if (is_leaf()) return ;

}


void DynamicBinaryNode::internals(vector<DynamicBinaryNode *>& v) {
	if (!is_leaf()) {
		v.push_back(this);
		m_nodes[LEFT]->externals(v);
		m_nodes[RIGHT]->externals(v);
	}

}

void DynamicBinaryNode::externals(vector<DynamicBinaryNode *>& v) {
	if (is_leaf()) {
		v.push_back(this);
	} else {
		m_nodes[LEFT]->internals(v);
		m_nodes[RIGHT]->internals(v);
	}
}

void DynamicBinaryNode::all_nodes(vector<DynamicBinaryNode *>& v) {
	if (is_leaf()) {
		v.push_back(this);
	} else {
		v.push_back(this);
		m_nodes[LEFT]->all_nodes(v);
		m_nodes[RIGHT]->all_nodes(v);
	}
}




