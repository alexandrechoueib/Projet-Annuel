#ifndef OBJECTS_ARRAY_H_
#define OBJECTS_ARRAY_H_

#include "essential/types.h"
#include "objects/object.h"
#include "essential/range.h"
#include "objects/container.h"

#include <vector>
using namespace ez::essential;

namespace ez {

namespace objects {

/*
   @CLASS
     The Array class which is a one-dimensional static container.
 */
template<class T>
class Array: public Object , public Container {
protected:

	// range of dimension 1 elements
	Range _range_x;

    // container of type vector
    std::vector <T> _data;
	
	//Iterator
	typedef T* iterator ;
public:
	/*
	   @WHAT
	    Constructor using integers
	 */
	Array(int, T);
    /*
	   @WHAT
	    Constructor using ranges
	 */
    Array(const Range&, T);

    /*
	   @WHAT
	    Copy constructor 
	 */
    Array(const Array& array);

    /*
	   @WHAT
	    Getter using dimension 1 index and dimension 2 index
	 */
    T get(int);

    /*
	   @WHAT
	    Setter using dimension 1 index and dimension 2 index and value
	 */
    void set(int, T);

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

	std::vector <T> data(){
		return _data;
	}
	/*
	   @WHAT
	    Destructor
	 */
	~Array() { };
};

} // end of namespace objects

} // end of namespace ez

#endif /* OBJECTS_ARRAY_H_ */
