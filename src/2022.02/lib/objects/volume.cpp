#include "objects/vector.h"
#include "objects/volume.h"

#include "essential/exception.h"
#include "extensions/algorithms.h"

namespace ezo = ez::objects;


template<class T>
T ezo::Volume<T>::get(int row , int column, int z) const{
    return _volume[row][column][z];
}   

template<class T>
void ezo::Volume<T>::fill(T value){
    ez::extensions::fill(this, value);
}

template<class T>
void ezo::Volume<T>::set(int row,int column, int z , T value){
    _volume[row][column][z] = value;
}