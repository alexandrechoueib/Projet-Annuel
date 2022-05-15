#ifndef OBJECTS_VECTOR_H_
#define OBJECTS_VECTOR_H_

#include <vector>

#include "objects/object.h"
#include "essential/range.h"
#include "objects/container.h"


namespace ezo = ez::objects;

namespace ez {

    namespace objects{
        
        template<class T>
        class Vector : public Object , public Container{
            private:
                Range _range_x;
                std::vector<T> _vector;

            protected:
                boolean isOutOfRange();

            public :
                typedef Vector self;
                typedef T * iterator;

                Vector();
                Vector(const Range& r):Object(),_range_x(r.first_value());
                Vector(const Vector& vector);
                T get(int i) throw Exception;
                void set(int i, T value);
                T get(int i);
                void push_back(T value);
                void delete(int position);

                iterator begin() { return _vector.begin() };
                iterator end() { return _vector.end() };
                //Virtual method from Object
                std::ostream& print(std::ostream& stream);
                integer compare(const Object& y);
                Object *clone();

                friend boolean operator==(const Vector &obj1, const Vector &obj2);
                friend boolean operator!=(const Vector &obj1, const Vector &obj2);
        };       
    }
}

#endif /* OBJECTS_INTEGER_H_ */
