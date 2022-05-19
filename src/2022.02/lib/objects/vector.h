#ifndef OBJECTS_VECTOR_H_
#define OBJECTS_VECTOR_H_

#include <vector>

#include "objects/object.h"
#include "essential/range.h"
#include "objects/container.h"

using namespace ez::essential;
using namespace ez::objects;


namespace ez {

    namespace objects{
        
        template<class T>
        class Vector : public Object , public Container{
            private:
                Range _range;
                std::vector<T> _vector;
                
            public :
                typedef T * iterator;

                Vector():_range(1,20){};
                Vector(const Range& r):_range(r){};
                Vector(Vector<T>& Vector): _range(1,20) {
                    _vector = Vector.getVector();
                    _size = _vector.size();
                }

                T get(int i) const ;
                void fill(T value);
                void set(int i, T value);
                void push_back(T value);
                void delete_value(int position);


                //Algorithm method implemente
                T find(T value) const;

                std::vector<T> getVector(){ return _vector ;};

                //iterator
                iterator begin() const{ return _vector.begin() ;};
                iterator end() const { return _vector.end() ;};
                
                //Virtual method from Object
                std::ostream& print (std::ostream& stream) const;
                integer compare(const Object& y) const;
                Object* clone();

                bool operator==(const Vector<T> &obj2) const  ;
                bool operator!=(const Vector<T> &obj2) const  ;
        };       
    }
}

#endif /* OBJECTS_VECTOR_H_ */
