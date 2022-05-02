/*
 * variable.h
 *
 *  Created on: Jun 19, 2018
 *      Author: richer
 */

#ifndef SRC_VERSION_2018_06_CUME_VARIABLE_H_
#define SRC_VERSION_2018_06_CUME_VARIABLE_H_

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Definition of a generic variable that handles data in host and
// device memory. Use the push() method to send value of variable
// from CPU to GPU and pull() method to get value from GPU and send
// it to CPU.
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include <numeric>

#include "base.h"

namespace ez {

namespace cume {

template<class T>
class Variable {
public:
	typedef Variable<T> self;

	T m_cpu_value;
	T *m_gpu_value;

	/**
	 * Default constructor with no parameter
	 */
	Variable() {
		cume_new_var(m_gpu_value, T);
	}

	/**
	 * Constructor given initial value of variable
	 * @param value value used to initialize variable
	 */
	Variable(T value) {
		m_cpu_value = value;
		cume_new_var(m_gpu_value, T);
	}

	Variable(const self& object) {
		m_cpu_value = object.m_cpu_value;
		cume_new_var(m_gpu_value, T);
	}

	virtual ~Variable() {
		cume_free(m_cpu_value);
	}

	self& operator=(const self& object) {
		if (&object != this) {
			cume_free(m_cpu_value);
			m_cpu_value = object.m_cpu_value;
			cume_new_var(m_gpu_value, T);
		}
		return *this;
	}

	T& value() {
		return m_cpu_value;
	}

	void value(T value) {
		m_cpu_value = value;
	}

	void push() {
		cume_push(m_gpu_data, m_cpu_data, T, 1);
	}

	void pull() {
		cume_pull(m_cpu_data, m_gpu_data, T, 1);
	}


	ostream& print(ostream& out) {
		out << m_cpu_value;
		return out;
	}

	friend ostream& operator<<(ostream& out, self& object) {
		return object.print(out);
	}
};

} // end of namespace cume

} // end of namespace ez

#endif /* SRC_VERSION_2018_06_CUME_VARIABLE_H_ */
