/*
 * prime_numbers.cpp
 *
 *  Created on: May 9, 2017
 *      Author: richer
 */

#include "arguments/import.h"
#include "essential/import.h"
#include "objects/import.h"
#include "extensions/import.h"
 
/**
 * Program entry.
 * We can specify the number of prime numbers to look for using
 * command line argument as "--max_primes=5" or "-m 5"
 */
int main(int argc, char *argv[]) {
	eza::ArgumentParser parser("prime_numbers", "compute prime numbers", argc, argv);
	natural max_primes = 20;

	try {
		parser.add_natural("max_primes", 'm', &max_primes, 
			"maximum number of prime numbers to find");
		parser.parse();
	} catch(Exception& e) {
		parser.report_error(e);
		return EXIT_FAILURE;
	}

	// declare vector of integer
	ezo::Vector<integer> prime_numbers;

	// look for prime numbers
	integer n = 1;
	while (prime_numbers.size() != max_primes) {
		if (ezo::Integer::is_prime(n)) prime_numbers << n;
		++n;
	}

	// print results
	cout << "The first " << max_primes << " prime numbers are:" << endl;
	cout << prime_numbers << endl;
	cout << "sum=" << ezx::sum(prime_numbers,0) << endl;
	
	// first 50 prime numbers
	ezo::Vector<integer> all_primes({
		  2,      3,      5,      7,     11,     13,     17,     19,     23,     29, 
		 31,     37,     41,     43,     47,     53,     59,     61,     67,     71,
		 73,     79,     83,     89,     97,    101,    103,    107,    109,    113, 
		127,    131,    137,    139,    149,    151,    157,    163,    167,    173,
		179,    181,    191,    193,    197,    199,    211,    223,    227,    229
    }); 

	natural maxi = max_primes;
	if (maxi > all_primes.size()) maxi = all_primes.size();

	cout << "solutions found and expected solutions are equal? ";
	if (ezx::are_equal(prime_numbers, all_primes, maxi)) {
		cout << "yes" << endl;
	} else {
		cout << "no" << endl;
	}

	for (natural i : eze::Range(1, maxi)) {
		if (prime_numbers[i] != all_primes[i]) {
			cout << "error: prime number at position " << i << " should be " << all_primes[i] << endl;
		}
	}    
	return EXIT_SUCCESS;
}


