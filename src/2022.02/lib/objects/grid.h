#ifndef OBJECTS_GRID_H_
#define OBJECTS_GRID_H_

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
     The Grid class which is a two-dimensional static container.
 */
template<class T>
class Grid: public Object , public Container {
protected:

	// range of dimension 1 elements
	Range _range_x;
    // range of dimension 2 elements
    Range _range_y;

    // container of type vector
    std::vector<std::vector <T>> _data;
	
	//Iterator
	typedef T* iterator ;
public:
	/*
	   @WHAT
	    Constructor using integers
	 */
	Grid(int, int, T);
    /*
	   @WHAT
	    Constructor using ranges
	 */
    Grid(const Range&, const Range&, T);

    /*
	   @WHAT
	    Copy constructor 
	 */
    Grid(const Grid& grid);

    /*
	   @WHAT
	    Getter using dimension 1 index and dimension 2 index
	 */
    T get(int, int);

    /*
	   @WHAT
	    Setter using dimension 1 index and dimension 2 index and value
	 */
    void set(int, int, T);

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
	std::vector<std::vector <T>> data(){
		return _data;
	}
	/*
	   @WHAT
	    Destructor
	 */
	~Grid() { };
};

} // end of namespace objects

} // end of namespace ez

#endif /* OBJECTS_GRID_H_ */
