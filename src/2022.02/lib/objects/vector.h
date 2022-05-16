#ifndef OBJECTS_VECTOR_H_
#define OBJECTS_VECTOR_H_

#include <vector>

#include "objects/object.h"
#include "essential/range.h"
#include "objects/container.h"


namespace ez {

    namespace objects{
        
        template<class T>
        class Vector : public Object , public Container{
            private:
                Range _range_x;
                std::vector<T> _vector;

            public :
                typedef Vector self;
                typedef T * iterator;

                Vector():_range_x(1);
                Vector(const Range& r):_range_x(r.first_value());
                Vector<T>(const Vector<T>& vector){
                    _vector = vector;
                    
                }

                T get(int i) const;
                void fill(int value);
                void set(int i, T value);
                void push_back(T value);
                void delete_value(int position);


                //Algorithm method implemente
                T find(T value) const;

                //iterator
                iterator begin() const{ return _vector.begin() };
                iterator end() const { return _vector.end() };
                
                //Virtual method from Object
                std::ostream& print (std::ostream& stream);
                integer compare(const Object& y);
                Object *clone();

                friend boolean operator==(const Vector<T> &obj1, const Vector<T> &obj2) const ;
                friend boolean operator!=(const Vector<T> &obj1, const Vector<T> &obj2) const ;
        };       
    }
}

#endif /* OBJECTS_VECTOR_H_ */
