 #ifndef OBJECTS_VOLUME_H
 #define OBJECTS_VOLUME_H

#include "objects/vector.h"


namespace ez {
    namespace objects{
        
        template<class T>
        class Volume : public Object, public Container {
            private:
                Range range_x;
                Range range_y;
                Range range_z;
                ez::objects::Vector<ez::objects::Vector<ez::objects::Vector<T>>> _volume;

            public:
                Volume<T>();
                Volume<T>(const Range &r);
                Volume<T>(const Volume<T> &volume);
                
                T get(int row, int column, int z) const;
                void fill(T value);
                void set(int row,int column, int z, T value);
                void push_back(T value);
                void delete_column(int position);
                void delete_row(int position);
                void delete_z(int position);

        };
    }
}

#endif /* OBJECTS_VOLUME_H_ */