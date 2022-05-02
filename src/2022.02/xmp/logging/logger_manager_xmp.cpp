#include "logging/import.h"
using namespace ezl; 

void proc() {
	LoggerManager::instance().attach(new ConsoleLogger("mystdout", &std::cout));
	Logger& log1 = LoggerManager::instance().get("mystdout");
	
	log1 << "Hello world" << endl;
	
	LoggerManager::instance().attach(new FileLogger("myfile", "file_logger.txt", FileLogger::TRUNCATE));
	Logger& log2 = LoggerManager::instance().get("myfile");
	
	for (int i = 0; i < 5; ++i) {
		log2 << "Hello world" << endl;
		log2 << "Bonjour monde" << endl;
	}

}

int main() { 
	proc();
	
	return EXIT_SUCCESS;
}

