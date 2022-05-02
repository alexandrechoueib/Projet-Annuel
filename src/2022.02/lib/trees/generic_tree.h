/*
 * generic_tree.h
 *
 *  Created on: Sep 6, 2017
 *      Author: richer
 */

#ifndef VERSION_2017_04_TREES_GENERIC_TREE_H_
#define VERSION_2017_04_TREES_GENERIC_TREE_H_

namespace ez {

namespace trees {

class GenericTree {
public:
	GenericTree() { }
	virtual ~GenericTree() { }

	virtual void print(std::ostream& out) = 0;

	friend std::ostream& operator<<(std::ostream& out, DynamicTree& obj) {
		obj.print(out);
		return out;
	}
};

} // end of namespace trees

} // end of namespace ez



#endif /* VERSION_2017_04_TREES_GENERIC_TREE_H_ */
