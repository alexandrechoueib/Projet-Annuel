/*
 * dynamic_tree.cpp
 *
 *  Created on: Sep 5, 2017
 *      Author: richer
 */

#include "../../version_2019.08/trees/dynamic_tree.h"

using namespace ez::trees;

#define parent(N) N->m_parent

DynamicTree::DynamicTree() : GenericTree(), m_root(nullptr) {

}

DynamicTree::DynamicTree(Node root) : GenericTree(), m_root(root) {

}

DynamicTree::DynamicTree(const DynamicTree& obj) : GenericTree() {
	m_root = obj.m_root->clone();
}

DynamicTree& DynamicTree::operator=(const DynamicTree& obj) {
	if (&obj != this) {
		if (m_root != nullptr) {
			m_root->clean();
			delete m_root;
		}
		m_root = (m_root == nullptr) ? nullptr : m_root->clone();
	}
	return *this;
}

DynamicTree *DynamicTree::clone() {
	return new DynamicTree((m_root == nullptr) ? nullptr : m_root->clone());
}

DynamicTree::Node DynamicTree::root(Node root) {
	Node old_root = m_root;
	m_root = root;
	return old_root;
}


void DynamicTree::print(std::ostream& out) {
	if (m_root == nullptr) {
		out << "nullptr";
	} else {
		m_root->print(out);
	}
}

void DynamicTree::internals(vector<DynamicTree::Node>& v) {
	if (m_root == nullptr) return ;
	m_root->internals(v);
}


void DynamicTree::externals(vector<DynamicTree::Node>& v) {
	if (m_root == nullptr) return ;
	m_root->externals(v);
}

void DynamicTree::all_nodes(vector<DynamicTree::Node>& v) {
	if (m_root == nullptr) return ;
	m_root->all_nodes(v);
}
