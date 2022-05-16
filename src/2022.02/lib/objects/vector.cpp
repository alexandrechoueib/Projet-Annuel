#include "objects/vector.h"
#include "essential/exception.h"
#include "extensions/algorithms.h"

namespace ezo = ez::objects;


template<class T>
T ezo::Vector<T>::get(int i){
    int index = _range_x.to_index(i);
    return _vector[index];
}

template<class T>
void ezo::Vector<T>::set(int i,T value){
    int index = _range_x.to_index(i);
    _vector[index] = value;
}
template<class T>
T ezo::Vector<T>::get(int i) const{
    int index = difference(_range,i)+1;
    return _vector[index];
}
template<class T>
void ezo::Vector<T>::push_back(T value){
    _vector.push_back(T);
    _size++; 
}

template<class T>
void ezo::Vector<T>::fill(int value){
    ez::extensions::fill(_vector, value);
}

/*
template<class T>
bool ezo::Vector<T>::all_diff() const{
    return ez::extensions::all_diff(_vector,compare())
}*/


template<class T>
void ezo::Vector<T>::delete_value(int position){
    for(int i=position ; i < _vector.size() ; i++){
        _vector[i++] = _vector[i];
    }
    _size--;
}


template<class T>
boolean operator==(const ezo::Vector<T> &obj1, const ezo::Vector<T> &obj2) const{
     if(ojb1.size() = obj2.size()) return false;

     for(unsigned int i=0 ; i < obj1.size() ; i++){
         if(obj1[i] != obj2[i]) 
            return false;
     }
     return true;
}
template<class T>
boolean operator!=(const ezo::Vector<T> &obj1, const ezo::Vector<T> &obj2) const{
    return !(obj1 == obj2);
}

template<class T>
ezo::Object *clone(){
    return new ezo::Vector<T>(*this);
}


