/*
 * test_sum.cpp
 *
 *  Created on: May 8, 2017
 *      Author: richer
 */

#include <gtest/gtest.h>
#include "objects/vector.h"
#include "extensions/algorithms.h"
#include "extensions/sum.h"
#include "extensions/getter.h"
#include <vector>
#include <numeric>
#include "essential/import.h"

using namespace std;
using namespace ez;


namespace ezo = ez::objects;
namespace ezx = ez::extensions;

class Person {
public:
	eze::text name;
	eze::real salary;

public:
	Person() : name("x"), salary(0) {

	}
	Person(text _name, real _salary) : name(_name), salary(_salary) {

	}
	~Person() {

	}
	Person(const Person& obj) {
		name = obj.name;
		salary = obj.salary;
	}
	Person& operator=(const Person& obj) {
		if (&obj != this) {
			name = obj.name;
			salary = obj.salary;
		}
		return *this;
	}
	eze::text get_name() {
		return name;
	}
	eze::real get_salary() {
		return salary;
	}

	// need to define this

	friend ostream& operator<<(ostream& out, Person& p) {
		out << p.get_name() << " " << p.get_salary();
		return out;
	}

	friend istream& operator>>(istream& in, Person& p) {
		in >> p.name >> p.salary;
		return in;
	}

	friend bool operator==(const Person& x, const Person& y) {
		return true;
	}

	/**
	 * Overloading of not equal operator
	 */
	friend bool operator!=(const Person& x, const Person& y) {
		return true;
	}

	/**
	 * overloading of less than operator used by the sort algorithm
	 */
	friend bool operator<(const Person& x, const Person& y) {
		return true;
	}

	Person& operator++() {
		++salary;
		return *this;
	}

	Person operator++(int junk) {
		Person ret(*this);
		++salary;
		return ret;
	}

	Person& operator+=(const Person& y) {
		return *this;
	}
};

class PersonSalary : public ezx::Getter<Person,eze::real> {
public:
	PersonSalary() {

	}
	eze::real get(const Person& p) {
		return const_cast<Person&>(p).get_salary();
	}

	eze::real get(Person *p) {
		return p->get_salary();
	}

};

TEST(TestExtensions, sum) {
	ezo::Vector<eze::integer> v(100);
	ezx::fill(v, 1);
	eze::integer v_result = ez::extensions::sum(v,0);
	EXPECT_EQ(static_cast<eze::natural>(v_result), v.size());

	ezo::Vector<ezo::Integer> w(100);
	ezx::fill(w, ezo::Integer(1));
	ezo::Integer w_result = ez::extensions::sum(w,ezo::Integer::zero_object);
	EXPECT_EQ(static_cast<eze::natural>(w_result.value()), w.size());


	ezo::Vector<Person> q(10);

	for (natural i = 1; i <= q.size(); ++i) {
		std::ostringstream oss;
		oss << "name" << i;
		q[i] = Person(oss.str(), 10 * i);
	}

	PersonSalary g;
	real q_result = ezx::sum(q, g, ezo::Real::zero);
	EXPECT_EQ(q_result, 550);

	cout << "q_result=" << q_result << endl;
	cout << q << endl;
}

int main(int argc, char *argv[]) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}



