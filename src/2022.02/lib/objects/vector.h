#ifndef OBJECTS_VECTOR_H_
#define OBJECTS_VECTOR_H_

#include "objects/object.h"

namespace ezo = ez::objects;

namespace ez {

    namespace objects{
        template<class T>
        class Vector : public Object {
            private:
                Range _range_x;
                std::vector<T> _vector;

            public :
                typedef Vector self;
                Vector():_range_x(1);
                Vector(const Range& r):_range_x(r.first_value());
                Vector(const Vector& vector);
                T get(int i);
                void set(int i, T value);
                void push_back(T value);
                void delete(int position);
                T pop(int position);
                
        }
        
    }
}

#endif /* OBJECTS_INTEGER_H_ */
