 #ifndef OBJECTS_MATRIX_H
 #define OBJECTS_MATRIX_H

#include "objects/vector.h"


namespace ez {
    namespace objects{
        
        template<class T>
        class Matrix : public Object, public Container {
            private:
                ez::objects::Vector<ez::objects::Vector<T>> _matrix;

            public:
                Matrix<T>();
                Matrix<T>(const Range &r);
                Matrix<T>(const Matrix<T> &matrix);
                
                T get(int i) const;
                void fill(int value);
                void set(int i, T value);
                void push_back(T value);
                void delete_column(int position);
                void delete_row(int position);


                friend boolean operator==(const Matrix<T> &obj1, const Matrix<T> &obj2) const ;
                friend boolean operator!=(const Matrix<T> &obj1, const Matrix<T> &obj2) const ;
        }
    }
}

#endif /* OBJECTS_MATRIX_H_ */