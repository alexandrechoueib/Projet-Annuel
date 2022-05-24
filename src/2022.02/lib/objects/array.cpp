#include "objects/array.h"
#include "essential/exception.h"
#include "essential/range.h"
#include "objects/container.h"

namespace ezo = ez::objects;


template<class T>
ezo::Array<T>::Array(int size_x, T init_value) {
    _range_x = new Range(1, size_x);
    _size = size_x;
    _data = new std::vector<T>(size_x);
}

template<class T>
ezo::Array<T>::Array(const ez::essential::Range& range_x, T init_value) {
    _range_x = range_x; 
    int size_x = _range_x.size();
    _size = size_x ;
    _data = new std::vector<T>(size_x);
}

template<class T>
ezo::Array<T>::Array(const ezo::Array<T>& array_copy) {
    _range_x = array_copy.range_x();
    _size = const_cast<Array&>(array_copy).size();
    _data = array_copy._data();
}

template<class T> 
T ezo::Array<T>::get(int i) {
    int index_x = _range_x.to_index(i);
    return _data[index_x];
}

template<class T>
void ezo::Array<T>::set(int i,  T value) {
    int index_x = _range_x.to_index(i);
    _data[index_x] = value;
}