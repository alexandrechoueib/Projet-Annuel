#include "objects/vector.h"
#include "essential/exception.h"


ezo::Vector():Object(),_range_x(1){

}

ezo::Vector(const Range& r):Object(),_range_x(r.first_value()){

}

ezo::Vector<T>(const Vector<T>& vector):Object(){
    _vector = vector;
    _size = vector.size();
    
}
boolean isOutOfRange(int i){ //modifier cette methode avec le range et lever une exception
    return xrange.to_index(i) ;
}
int difference(int a, int b){
    if( a > b) {
        int tmp = a ;
        a = b
        b = tmp;
    }
    //a est forcement plus petit que b
    if(a < 0 && b < 0)
        return (b - a); // -3 - (-5 ) = 2
    else if(a < 0 && b > 0)
        return abs(a-b);
    else if(a > 0 && b > 0)
        return (b - a);
}

T ezo::Vector<T>::get(int i) throw Exception{
    if(isOutOfRange(i)) {
         // lancer une exception
        notify( "out of range i :" + i );
    }
    return _vector[i];
}

void ezo::Vector<T>::set(int i,T value){
    if(isOutOfRange(i)) return;
    int index = difference(_range,i) + 1;
    vector[index] = value;
}

T ezo::Vector<T>::get(int i){
    int index = difference(_range,i)+1;
    return vector[index];
}

void ezo::Vector<T>::push_back(T value){
    _vector.push_back(T);
    _size += 1; 
}

void ezo::Vector<T>::delete(int position){
    for(int i=position ; i < _vector.size() ; i++){
        _vector[i++] = _vector[i];
    }
}



boolean operator==(const Vector &obj1, const Vector &obj2){
     if(ojb1.size() = obj2.size()) return false;

     for(unsigned int i=0 ; i < obj1.size() ; i++){
         if(obj1[i] != obj2[i]) 
            return false;
     }
     return true;
}

boolean operator!=(const Vector &obj1, const Vector &obj2){
    return !(obj1 == obj2);
}

ezo::Object *clone(){
    return new Vector(*this);
}


