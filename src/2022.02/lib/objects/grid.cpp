#include "objects/grid.h"
#include "essential/exception.h"
#include "essential/range.h"
#include "objects/container.h"

namespace ezo = ez::objects;


template<class T>
ezo::Grid<T>::Grid(int size_x, int size_y, T init_value) {
    _range_x = new Range(1, size_x);
    _range_y = new Range(1, size_y);
    _size = size_x * size_y;
    _data = new std::vector<std::vector<T>>(size_x, std::vector<T>(size_y, init_value));
}

template<class T>
ezo::Grid<T>::Grid(const ez::essential::Range& range_x, const ez::essential::Range& range_y, T init_value) {
    _range_x = range_x; 
    _range_y = range_y; 
    int size_x = _range_x.size();
    int size_y = _range_y.size();
    _size = size_x * size_y;
    _data = new std::vector<std::vector<T>>(size_x, std::vector<T>(size_y, init_value));
}

template<class T>
ezo::Grid<T>::Grid(const ezo::Grid<T>& grid_copy) {
    _range_x = grid_copy.range_x();
    _range_y = grid_copy.range_y();
    _size = const_cast<Grid&>(grid_copy).size();
    _data = grid_copy._data();
}

template<class T> 
T ezo::Grid<T>::get(int i, int j) {
    int index_x = _range_x.to_index(i);
    int index_y = _range_y.to_index(j);
    return _data[index_x][index_y];
}

template<class T>
void ezo::Grid<T>::set(int i, int j, T value) {
    int index_x = _range_x.to_index(i);
    int index_y = _range_y.to_index(j);
    _data[index_x][index_y] = value;
}