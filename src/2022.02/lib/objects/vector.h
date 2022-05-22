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

                Vector();
                Vector(const Range& r);
                Vector(Vector<T>& Vector);

                T get(int) const ;
                void fill(T);
                void set(int , T);
                void push_back(T);
                void delete_value(int);


                //Algorithm method implemente
                T find(T) const;

                std::vector<T> getVector(){ return _vector ;};

                //iterator
                iterator begin() const{ return _vector.begin() ;};
                iterator end() const { return _vector.end() ;};
                
                //Virtual method from Object
                std::ostream& print (std::ostream& stream) const override;
                integer compare(const Object& y) const override;
                Object* clone() override;

                bool operator==(const Vector<T> &obj2) const  ;
                bool operator!=(const Vector<T> &obj2) const  ;

                ~Vector() ;
        };       
    }
}

#endif /* OBJECTS_VECTOR_H_ */
