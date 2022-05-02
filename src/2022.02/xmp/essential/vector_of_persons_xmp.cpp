/*
 * vector_of_persons.cpp
 *
 *  Created on: Jul 31, 2017
 *      Author: Jean-Michel Richer
 *
 *  In this example we are using a Vector of Person(s).
 *  We show how to use
 *  <ul>
 *  	<li>the sort capability</li>
 *  	<li>the extraction of the field 'm_age' of a Person
 *  		in order to compute the sum of ages</li>
 *  </ul>
 *
 *  Use of ez::arguments, Vector of ez::objects and ez::extensions
 */

#include "person.h"
#include "arguments/import.h"
#include "essential/import.h"
#include "objects/import.h"
#include "extensions/import.h"

/**
 * Comparator function used to compare two Person
 * in function of their age
 */
bool AgeComparator( const Person& p, const Person& q ) {

	return (p._age < q._age);
	
}

/*
  @CLASS 
  	Class used to extract age from a Person
 */
class PersonAgeGetter : public ezx::Getter<Person, integer> {
public:
	/*
	  @WHAT
	    Constructor
	 */
	PersonAgeGetter() : Getter<Person, integer>() { 
	}
	
	/*
	   @WHAT
	     return age from reference to Person
	 */
	integer get( const Person& var ) { 
	
		return var._age; 
		
	}
	
	/*
	  @WHAT
	    return age from pointer to Person
	 */   
	integer get( Person *var ) { 
	
		return var->_age; 
		
	}
};

/*
  @WHAT
    Perform treatments on a Vector of Persons
  
 */
void vector_of_persons() {

	cout << eze::Terminal::line_type_3 << endl;
	cout << "Vector of persons" << endl;
	cout << eze::Terminal::line_type_3 << endl;

	// define vector of persons
	ezo::Vector<Person> vp;

	// populate vector
	vp.insert_last( Person( "toto", 10 ) );
	vp.insert_last( Person( "riri", 11 ) );
	vp.insert_last( Person( "fifi", 10 ) );
	vp.insert_last( Person( "picsou", 60 ) );
	vp.insert_last( Person( "picsou", 9 ) );
	vp.insert_last( Person( "picsou", 5 ) );

	cout << "default values=" << vp << endl;

	ezx::sort(vp);
	cout << "sort using person.compare (ascending)=" << vp << endl;

	ezx::rsort(vp);
	cout << "sort using person.compare (descending)=" <<vp << endl;

	ezx::rsort_comparator(vp, AgeComparator);
	cout << "sort using AgeComparator=" <<vp << endl;

	// use extensions to compute sum of ages of the persons
	PersonAgeGetter pag;
	integer sum = ezx::sum( vp, pag, 0 );
	cout << "sum of ages=" << sum << endl;
}

/**
 * Comparator function used to compare two Person
 * in function of their age
 */
bool PtrPersonComparator(Person *p, Person *q) {

	return p->compare(*q) < 0;
	
}

bool PtrAgeComparator(Person *p, Person *q) {

	return (p->_age < q->_age);
	
}

/*
  @WHAT
    Perform treatments on a Vector of pointers to Persons
  
 */
void vector_of_pointer_of_persons() {

	cout << eze::Terminal::line_type_3 << endl;
	cout << "Vector of pointers of persons" << endl;
	cout << eze::Terminal::line_type_3 << endl;

	// define vector of pointer of persons
	ezo::Vector<Person *> vp;

	// fill vector
	vp.insert_last( new Person( "toto", 10 ) );
	vp.insert_last( new Person( "riri", 11 ) );
	vp.insert_last( new Person( "fifi", 10 ) );
	vp.insert_last( new Person( "picsou", 60 ) );
	vp.insert_last( new Person( "picsou", 9 ) );
	vp.insert_last( new Person( "picsou", 5 ) );

	cout << "default values=" << vp << endl;


	ezx::sort_comparator( vp, PtrPersonComparator );
	cout << "sort using PtrPersonComparator (ascending)=" << vp << endl;


	ezx::sort( vp );
	cout << "sort using PtrPersonComparator (descending)=" <<vp << endl;


	ezx::sort_comparator( vp, PtrAgeComparator );
	cout << "sort using PtrAgeComparator=" <<vp << endl;

	// use extensions to compute sum of ages of the persons
	PersonAgeGetter pag;
	integer sum = ezx::sum( vp, pag, 0 );
	cout << "sum of ages=" << sum << endl;

}

/*
  @WHAT
    Main function
 */
int main(int argc, char *argv[]) {

	vector_of_persons();
	vector_of_pointer_of_persons();

	return EXIT_SUCCESS;

}
