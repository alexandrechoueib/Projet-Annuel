/*
 * text.cpp
 *
 *  Created on: Apr 11, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "objects/text.h"

using namespace ez::objects;

std::ostream& Text::print(std::ostream& stream) {
	stream << "\"" << m_value << "\"";
	return stream;
}

void Text::trim_right(const text& pattern) {
	TextUtils::trim_right(m_value, pattern);
}

void Text::trim_left(const text& pattern) {
	TextUtils::trim_left(m_value, pattern);
}

void Text::trim(const text& pattern) {
	Text::trim_right();
	Text::trim_left();
}

void Text::erase(const text& pattern) {
	TextUtils::erase(m_value, pattern);
}

void Text::remove_spaces() {
	TextUtils::remove_spaces(m_value);
}

void Text::lower() {
	TextUtils::lower(m_value);
}

void Text::upper() {
	TextUtils::upper(m_value);
}

void Text::explode(vector<text>& v, const text& delim) {
	TextUtils::explode(m_value, v, delim);
}

void Text::implode(vector<text>& v, const text& delim) {
	TextUtils::implode(m_value, v, delim);
}

void Text::implode(vector<char>& v, const text& delim) {
	TextUtils::implode(m_value, v, delim);
}

bool Text::starts_with(text& st_str) {
	return TextUtils::starts_with(m_value, st_str);
}

bool Text::starts_with(char *st_str) {
	return TextUtils::starts_with(m_value, st_str);
}

bool Text::starts_with(Text& st_str) {
	return TextUtils::starts_with(m_value, st_str.m_value);
}

bool Text::ends_with(text& st_str, bool rm_flag) {
	return TextUtils::ends_with(m_value, st_str, rm_flag);
}

bool Text::ends_with(char *st_str, bool rm_flag) {
	return TextUtils::ends_with(m_value, st_str, rm_flag);
}

bool Text::ends_with(Text& st_str, bool rm_flag) {
	return TextUtils::ends_with(m_value, st_str.m_value, rm_flag);
}


void Text::replace_char(char src, char repl) {
	TextUtils::replace_char(m_value, src, repl);
}


size_t Text::position_of(text& st_str) {
	return TextUtils::position_of(m_value, st_str);
}

size_t Text::position_of(char *st_str) {
	return TextUtils::position_of(m_value, st_str);
}

void Text::remove_at(natural pos, natural length) {
	TextUtils::remove_at(m_value, pos, length);
}

string Text::_text_(boolean b) {
	return (b == false) ? "false" : "true";
}

string Text::_text_(character c) {
	return string(1, c);
}

string Text::_text_(integer i) {
	return std::to_string(i);
}

string Text::_text_(real r) {
	return std::to_string(r);
}
