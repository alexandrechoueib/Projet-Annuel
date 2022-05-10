#include "objects/vector.h"


ezo::Vector():Object(),_range_x(1){

}

ezo::Vector(const Range& r):Object(),_range_x(r.first_value()){

}

ezo::Vector<T>(const Vector<T>& vector):Object(){
    _vector = vector;
    _size = vector.size();
    
}
boolean isOutOfRange(int i){
    return (i+1 < range_x || i+1 > _size ) ? true : false;
 
}

T ezo::Vector<T>::get(int i){
    if(isOutOfRange(i)) return null;
    return _vector[i];
}

void ezo::Vector<T>::set(int i,T value){
    if(isOutOfRange(i)) return;
    
    for(int ii = 0 ; i < _size ; i++){
        if(ii == i+1){
            _vector[ii] = value;
        }
    }
}

void ezo::Vector<T>::push_back(T value){
    _vector.push_back(T);
    _size += 1; 
}

void ezo::Vector<T>::delete(int position){

}

T ezo::Vector<T>::pop(int position){

}
