/*
 * language.h
 *
 *  Created on: Aug 13, 2017
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

#ifndef ESSENTIAL_LANGUAGE_H_
#define ESSENTIAL_LANGUAGE_H_

#define on_do(x,y) if (x) y;

#define on_return(x,y) if (x) return y;

#define for_in(x,lo,hi) \
	for (auto x : ez::essential::Range(lo,hi))


#endif /* ESSENTIAL_LANGUAGE_H_ */

