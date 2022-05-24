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

template<class T>
void ezo::Matrix<T>::delete_column(int position){
    for (unsigned i = 0; i < _matrix.size(); ++i)
    {
        if (_matrix[i].size() > position)
        {
            _matrix[i].erase(_matrix[i].begin() + position);
        }
    }
}

template<class T>
void ezo::Matrix<T>::delete_row(int position){
    if (_matrix.size() > position)
        _matrix.erase( _matrix.begin() + position );   
}