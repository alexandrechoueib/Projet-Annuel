#include "objects/mesh.h"
#include "essential/exception.h"
#include "essential/range.h"

namespace ezo = ez::objects;


template<class T>
ezo::Mesh<T>::Mesh(int size_x, int size_y, int size_z, T init_value) {
    _range_x = new Range(1, size_x);
    _range_y = new Range(1, size_y);
    _range_z = new Range(1, size_z);
    _size = size_x * size_y * size_z;
    _data = new std::vector<std::vector<std::vector<T>>>(size_x, std::vector<std::vector<T>>(size_y, std::vector<T>(size_z, init_value)));
}

template<class T>
ezo::Mesh<T>::Mesh(const ez::essential::Range& range_x, const ez::essential::Range& range_y, const ez::essential::Range& range_z, T init_value) {
    _range_x = range_x; 
    _range_y = range_y; 
    _range_z = range_z; 
    int size_x = _range_x.size();
    int size_y = _range_y.size();
    int size_z = _range_z.size();
    _size = size_x * size_y * size_z;
    _data = new std::vector<std::vector<std::vector<T>>>(size_x, std::vector<std::vector<T>>(size_y, std::vector<T>(size_z, init_value)));
}

template<class T>
ezo::Mesh<T>::Mesh(const ezo::Mesh<T>& mesh_copy) {
    _range_x = mesh_copy.range_x();
    _range_y = mesh_copy.range_y();
    _range_z = mesh_copy.range_z();
    _size = const_cast<Mesh&>(mesh_copy).size();
    _data = mesh_copy._data();
}

template<class T> 
T ezo::Mesh<T>::get(int i, int j, int k) {
    int index_x = _range_x.to_index(i);
    int index_y = _range_y.to_index(j);
    int index_z = _range_z.to_index(k);
    return _data[index_x][index_y][index_z];
}

template<class T>
void ezo::Mesh<T>::set(int i, int j, int k, T value) {
    int index_x = _range_x.to_index(i);
    int index_y = _range_y.to_index(j);
    int index_z = _range_y.to_index(k);
    _data[index_x][index_y][index_z] = value;
}