/*
 * logger_mamager.h
 *
 *  Created on: Jul 28, 2013
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

/*
    EZLib version 2022.02
    Copyright (C) 2019-2022  Jean-Michel Richer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

*/

#ifndef LOGGING_LOGGER_MAMAGER_H_
#define LOGGING_LOGGER_MAMAGER_H_

#include <map>

#include "logging/logger.h"


namespace ez {

namespace logging {

/**
 * Logger Manager for the simple logging package.
 * The role of the LoggerManager is to record loggers (@see Logger).
 * By default a "root logger" (which is a console logger)
 * is created.
 * When using a logger, end line to display information by sending
 * the '\n' character or Logger::endl.
 *
 * How to use simple logging system:
 * <ul>
 * <li>register new ConsoleLogger called <em>stdout</em>:
 * <pre><code>
 * LoggerManager::instance().attach(new ConsoleLogger("stdout", &std::cout));
 * </code></pre>
 * </li>
 * <li> use logger <em>stdout</em>:
 * <pre><code>
 * Logger& log1 = LoggerManager::instance().get("stdout");
 * log1 << "hello world\n";
 * log1 << "message" << Logger::endl;
 * </code></pre>
 * </li>
 * <li>register new FileLogger called <em>my</em>:
 * <pre><code>
 * LoggerManager::instance().attach(new FileLogger("my", "file_logger.txt",
 * FileLogger::TRUNCATE));
 * 	Logger& log2 = LoggerManager::instance().get("my");
 * 	</code></pre>
 * 	</li>
 * 	</ul>
 */
class LoggerManager {
public:
	/**
	 * return instance of this class
	 */
	static LoggerManager& instance();


	/**
	 * destructor that will remove all loggers
	 */
	virtual ~LoggerManager();


	/**
	 * add new logger, throw exception if a logger with same identifier
	 * already exists. In this case you should use the "get_logger" method
	 * @param l pointer to Logger
	 */
	void attach( Logger *l );


	/**
	 * remove logger and return pointer to it. An exception will be generated
	 * if the name of the logger is not found
	 * @param name identifier of Logger
	 * @return pointer to Logger
	 */
	Logger *detach( text name );


	/**
	 * return reference to existing logger identified by its name, an
	 * exception is raised if the logger is not found
	 * @param name identifier of Logger
	 * @param or_create string to create log (@see create)
	 * @return reference to logger if if exists or throws exception
	 */
	Logger& get( text name, text or_create="" );


	/**
	 * return reference to root logger
	 * @return reference to root logger
	 */
	Logger& root_logger();


	/**
	 * create logger from string description:
	 * <ul>
	 *   <li>for console logger use "console" or "console,stdout" or
	 *   "console,stderr"</li>
	 *   <li>for file logger use "file,name-of-file" or
	 *   "file,name-of-file,truncate"</li>
	 * </ul>
	 */
	Logger& create( text s );


protected:
	/**
	 * unique instance of LoggerManager
	 */
	static LoggerManager *_instance;
	/**
	 * unique instance of root logger
	 */
	static Logger *_root_logger;


	/**
	 * constructor that will create a root logger
	 */
	LoggerManager();

	/**
	 * records the loggers and their names
	 */
	map<text, Logger *> _loggers;

private:

	void explode(text& s, vector<text>& v);

};

} // end of namespace logging

} // end of namespace ez

#endif /* LOGGING_LOGGER_MAMAGER_H_ */
