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
                
                T get(int row, int column) const;
                void fill(T value);
                void set(int row,int column, T value);
                void push_back(T value);
                void delete_column(int position);
                void delete_row(int position);


                //friend boolean operator==(const Matrix<T> &obj1, const Matrix<T> &obj2)  ;
                //friend boolean operator!=(const Matrix<T> &obj1, const Matrix<T> &obj2)  ;
        };
    }
}

#endif /* OBJECTS_MATRIX_H_ */