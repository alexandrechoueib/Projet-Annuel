#ifndef OBJECTS_MESH_H_
#define OBJECTS_MESH_H_

#include "essential/types.h"
#include "essential/range.h"
#include "objects/container.h"
#include "objects/object.h"
#include <vector>
using namespace ez::essential;

namespace ez {

namespace objects {

/*
   @CLASS
     The Mesh class which is a third-dimensional static container.
 */
template<class T>
class Mesh: public Object, public Container {
protected:

	// range of dimension 1 elements
	Range _range_x;
    // range of dimension 2 elements
    Range _range_y;
    // range of dimension 3 elements
    Range _range_z;

    // container of type vector
    std::vector<std::vector<std::vector <T>>> _data;

	typedef T* iterator;
	
public:
	/*
	   @WHAT
	    Constructor using integers
	 */
	Mesh(int, int, int, T);

    /*
	   @WHAT
	    Constructor using ranges
	 */
    Mesh(const Range&, const Range&, const Range&, T);

    /*
	   @WHAT
	    Copy constructor 
	 */
    Mesh(const Mesh& mesh);

    /*
	   @WHAT
	    Getter using dimension 1 & 2 & 3 index
	 */
    T get(int, int, int);

    /*
	   @WHAT
	    Setter using dimension 1 index and dimension 2 index and value
	 */
    void set(int, int, int, T);

	/*
	   @WHAT
	    Iterators begin & end
	 */
	iterator begin() const { 
		return _data.begin() ;
	};
	iterator end() const { 
		return _data.end() ;
	};

	Range range_x() {
		return _range_x;
	}
	Range range_y() {
		return _range_y;
	}
    Range range_z() {
		return _range_y;
	}
	std::vector<std::vector<std::vector <T>>> data(){
		return _data;
	}
	/*
	   @WHAT
	    Destructor
	 */
	~Mesh() { }
};

} // end of namespace objects

} // end of namespace ez

#endif /* OBJECTS_CONTAINER_H_ */
