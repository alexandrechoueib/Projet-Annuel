/*
 * person.h
 *
 *  Created on: Jul 31, 2017
 *      Author: richer
 */

#ifndef PERSON_H_
#define PERSON_H_

#include "essential/import.h"
#include "objects/import.h"

using namespace eze;
using namespace ezo;

/*
   @CLASS
     Person with name and age
     
 */
class Person : public Object {
protected:
public:
	typedef Person self;
	
	// name of person
	text    _name;
	// age in years
	integer _age;

	/*
	  @WHAT
	     Default constructor
	 */
	Person();
	
	/*
	  @WHAT
	     Constructor with arguments
	     
	   @PARAMETERS
	     @param:name name of person
	     @param:age age in years
	 */
	Person( text name, integer age );
	
	/*
	  @WHAT
	  	Copy constructor
	 */
	Person( const self& obj );
	
	/*
	  @WHAT
	    Assignment operator
	 */
	self& operator=( const self& obj );
	
	/*
	  @WHAT
	    Destructor
	 */
	~Person();

	// ==========================================
	// you need to define the following methods
	// that are inherited from the class Object
	// if you have specific needs
	// - print to print object's contents
	// - output to serialize object
	// - input to unserialize object
	// - compare to compare two objects
	// - clone to return a copy of an object
	// ==========================================

	std::ostream& print( std::ostream& stream ) {
	
		stream << "(" << _name << "," << _age << ")";
		return stream;
		
	}


	/*
	   @WHAT
	    Compare two persons using their names and then
	    their ages
	    
	   @HOW
	     We first compare then names and then the ages
	     
	   @RETURN
	     @list
	      -1 if this Person is less than @param:y
	       0 if this Person is equal to @param:y
	       1 if this Person is greater than @param:y
	     @endlist    
	 */
	integer compare( const Object& y ) {
		Person& y_obj = *dynamic_cast<Person *>( &const_cast<Object&>( y ) );
		if (_name < y_obj._name) return -1;
		if (_name > y_obj._name) return 1;
		return _age - y_obj._age;
	}

	/**
	 * WHAT
	 *   Clone object
	 */
	Object *clone() {
		return new Person(*this);
	}



};



#endif /* PERSON_H_ */
