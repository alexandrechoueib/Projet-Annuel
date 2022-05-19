#include "objects/vector.h"
#include "objects/matrix.h"
#include "essential/exception.h"
#include "extensions/algorithms.h"

namespace ezo = ez::objects;


template<class T>
T ezo::Matrix<T>::get(int row , int column) const{
    return _matrix[row][column];
}   

template<class T>
void ezo::Matrix<T>::fill(T value){
    ez::extensions::fill(this, value);
}

template<class T>
void ezo::Matrix<T>::set(int row,int column, T value){
    _matrix[row][column] = value;
}