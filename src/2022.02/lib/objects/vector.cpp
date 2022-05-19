#include "objects/vector.h"
#include "essential/exception.h"
#include "extensions/algorithms.h"

namespace ezo = ez::objects;


template<class T>
void ezo::Vector<T>::set(int i,T value){
    int index = const_cast<Range&>(_range).to_index(i);
    _vector[index] = value;
}
template<class T>
T ezo::Vector<T>::get(int i) const{
    int index = const_cast<Range&>(_range).to_index(i);
    return _vector[index];
}
template<class T>
void ezo::Vector<T>::push_back(T value){
    _vector.push_back(value);
    _size++; 
}

template<class T>
void ezo::Vector<T>::fill(T value){
    ez::extensions::fill(_vector, value);
}


template<class T>
void ezo::Vector<T>::delete_value(int position){
    for(int i=position ; i < _vector.size() ; i++){
        _vector[i++] = _vector[i];
    }
    _size--;
}


template<class T>
bool ezo::Vector<T>::operator==(const ezo::Vector<T> &obj2) const {
    return this->_vector == obj2._vector ;
}
template<class T>
bool ezo::Vector<T>::operator!=(const ezo::Vector<T> &obj2) const {
    return !(this == obj2);
}

template<class T>
Object* ezo::Vector<T>::clone(){
    return new ezo::Vector<T>(*this);
}

template<class T>
std::ostream& ezo::Vector<T>::print (std::ostream& stream) const{
    stream << "size :"+ _size;
    stream << std::endl;
    stream << " vector : ";
    for(int i = const_cast<Range&>(_range).first_value(); i < const_cast<Range&>(_range).last_value() ; i++ ){
        stream << _vector[i] << " ";
    }
    stream << std::endl;

    return stream;
}

	
/**
* compare two objects
* @return 0 if objects are identical, negative value if this < y,
* positive value if this > y
*/
template<class T>
integer ezo::Vector<T>::compare(const Object& y) const{
    Vector<T>& y_obj = *dynamic_cast<Vector *>(&const_cast<Object&>(y));
    if(_vector == y_obj._vector) return 0;
    return _vector < y_obj._vector ? -1 : +1;
}


