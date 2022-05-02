/*
 * argument_parser.cpp
 *
 *  Created on: May 7, 2017
 *      Author: richer
 */

#include "arguments/argument_parser.h"

using namespace ez::essential;
using namespace ez::arguments;

static bool _help_flag = false;
static text _output_file_name;
static std::ofstream _output;
static std::streambuf *_oldbuffer = nullptr;


ArgumentParser::ArgumentParser( text program_name, text description, int argc, char *argv[] ) {

	_program_name = program_name;
	_program_desc = description;
	_help_command = add_flag( "help", 'h', &_help_flag, "help flag, print synopsis" );
	_output_command = add_text( "output", 'o', &_output_file_name, "output file, used to redirect output" );
	_show_synopsis = true;
	for (int i = 1; i < argc; ++i) _command.push_back( argv[i] );

}


ArgumentParser::ArgumentParser( text program_name, text description, std::vector<string>& command ) {

	_program_name = program_name;
	_program_desc = description;
	_help_command = add_flag( "help", 'h', &_help_flag, "help flag, print synopsis" );
	_output_command = add_text( "output", 'o', &_output_file_name, "output file, used to redirect output" );
	_show_synopsis = true;
	copy( command.begin(), command.end(), back_inserter( _command ) );

}


ArgumentParser::~ArgumentParser() {

	for (auto arg : _arguments) {
		delete arg;
	}
	
	_map_long.clear();
	_map_short.clear();
	
	if (_oldbuffer != nullptr) cout.rdbuf( _oldbuffer );
	
}


void ArgumentParser::clean() {

	for (auto a : _arguments) {
		delete a;
	}

}


Argument *ArgumentParser::add( Argument *arg ) {

	for (auto a : _arguments) {
		if (a->get_long_label() == arg->get_long_label()) {
			notify( "label " << a->get_long_label() << " already exists,"
					<< "choose another label" );
		}
		
		if (arg->get_short_label() != '\0') {
			if (a->get_short_label() == arg->get_short_label()) {
				notify( "label " << a->get_short_label() << " already exists,"
						<< "choose another label" );
			}
		}
	}
	
	_arguments.push_back( arg );
	_map_long[ arg->get_long_label() ] = arg;
	if (arg->get_short_label() != '\0') {
		_map_short[ arg->get_short_label() ] = arg;
	}
	
	return arg;
	
}


Argument *ArgumentParser::add_flag( text l_label, char s_label, bool *v, text descr, trigger_t tg ) {

	Argument *arg = new FlagArgument( l_label, s_label, v, descr, tg );
	return add( arg );
	
}


Argument *ArgumentParser::add_boolean( text l_label, char s_label, bool *v, text descr, trigger_t tg ) {

	Argument *arg = new BooleanArgument( l_label, s_label, v, descr, tg );
	return add( arg );

}


Argument *ArgumentParser::add_integer( text l_label, char s_label, integer *v, text descr, trigger_t tg ) {

	Argument *arg = new IntegerArgument( l_label, s_label, v, descr, tg );
	return add( arg );

}


Argument *ArgumentParser::add_natural( text l_label, char s_label, natural *v, text descr, trigger_t tg ) {

	Argument *arg = new NaturalArgument( l_label, s_label, v, descr, tg );
	return add( arg );

}


Argument *ArgumentParser::add_real( text l_label, char s_label, real *v, text descr, trigger_t tg ) {

	Argument *arg = new RealArgument( l_label, s_label, v, descr, tg );
	return add( arg );

}


Argument *ArgumentParser::add_real_range( text l_label, char s_label,  real *v,
		real mini, real maxi, text descr, trigger_t tg ) {

	Argument *arg = new RealRangeArgument( l_label, s_label, v, mini, maxi, descr, tg );
	return add( arg );

}


Argument *ArgumentParser::add_integer_range( text l_label, char s_label,  integer *v,
		integer mini, integer maxi, text descr, trigger_t tg ) {

	Argument *arg = new IntegerRangeArgument( l_label, s_label, v, mini, maxi, descr, tg );
	return add( arg );

}


Argument *ArgumentParser::add_natural_range( text l_label, char s_label,  natural *v,
		natural mini, natural maxi, text descr, trigger_t tg ) {
	Argument *arg = new NaturalRangeArgument( l_label, s_label, v, mini, maxi, descr, tg );
	return add( arg );
	
}


Argument *ArgumentParser::add_text( text l_label, char s_label, text *v, text descr, trigger_t tg ) {
	Argument *arg = new TextArgument( l_label, s_label, v, descr, tg );
	return add( arg );
}


Argument *ArgumentParser::add_options( text l_label, char s_label, natural *v, vector<text>& opt,
		text descr, trigger_t tg ) {
		
	Argument *arg = new OptionsArgument( l_label, s_label, v, opt, descr, tg );
	return add( arg );
	
}


void ArgumentParser::parse( std::vector<string>& remaining,
		natural max_remaining) {
		
	size_t i = 0;

	text long_option_intro = "--";
	text short_option_intro = "-";

	_show_synopsis = true;
	while (i < _command.size()) {
		text arg = _command[i];
		text value = "";

		if (ez::essential::TextUtils::starts_with( arg, long_option_intro )) {
			text label = "";

			ez::essential::TextUtils::trim_left( arg, short_option_intro );

			text s_equal = "=";
			size_t pos = ez::essential::TextUtils::position_of( arg, s_equal );

			if (pos > 0) {
				label = arg.substr( 0, pos-1 );
				if (pos + 1 <= arg.length()) {
					value = arg.substr( pos );
				}
			} else {
				label = arg;
			}

			Argument *argument = nullptr;
			auto it_long = _map_long.find( label );
			if (it_long != _map_long.end()) {
				argument = (*it_long).second;
			} else {
				notify( "label \"" + label + "\" does not exist" );
			}
			
			argument->parse( value );
			argument->set_found();

		} else if (ez::essential::TextUtils::starts_with( arg, short_option_intro )) {
			char label;

			ez::essential::TextUtils::trim_left( arg, short_option_intro );
			if (arg.length() != 1) {
				notify( "command line argument \"" << _command[ i ]
						<< "\" is in short format and needs only one character");
			}
			label = arg[0];
			Argument *argument = nullptr;
			
			auto it_short = _map_short.find(label);
			if (it_short != _map_short.end()) {
				argument = (*it_short).second;
			} else {
				notify( "label \"" << label << "\" does not exist" );
			}
			
			if (typeid(*argument) != typeid(FlagArgument)) {
				++i;
				if (i == _command.size()) {
					notify( "no value for argument -" << label );
				}
				value = _command[i];
				argument->parse( value );
				argument->set_found();
				
			} else {
				text empty = "";
				argument->parse(empty);
				argument->set_found();
				
			}
			
		} else {
			//notify("argument does not follow short or long format: \"" << m_argv[i]
			// << "\", use - or -- before argument label");
			remaining.push_back( arg );
		}
		++i;
	}

	if (_help_command->was_found()) {
		print_synopsis( std::cout );
		exit( EXIT_SUCCESS );
	}

	_show_synopsis = false;

	for (auto arg : _arguments) {
		if (arg->was_found()) {
			arg->call_trigger();
		}
	}

	_help_flag = false;

	// check that required command line options were parsed
	for (auto arg : _arguments) {
		if (arg->is_required()) {
			if (!arg->was_found()) {
				notify( "command line option \"" << arg->get_long_label() << "\" needs to be defined" );
			}
		}
	}

	// launch triggers
	_show_synopsis = false;
	for (auto arg : _arguments) {
		if (arg->was_found()) {
			arg->call_trigger();
		}
	}

	// redirects output if needed
	if (_output_file_name.length() > 0) {
		_oldbuffer = cout.rdbuf( _output.rdbuf() );
		_output.open( _output_file_name.c_str() );
		if (!_output) {
			notify( "could not open file \"" << _output_file_name << " \" for writing" );
		}
	}

	if (remaining.size() > max_remaining) {
		notify( "too many extra command line arguments not starting by '-' or '--'" );
	}

}


void ArgumentParser::print_synopsis_text( std::ostream& out, integer lvl, text s, natural length ) {

	natural l = 0;
	istringstream iss( s );
	text word;

	for (integer i = 0; i < lvl; ++i) out << "\t";
	while (iss >> word) {
		if ((l + word.size()) > length) {
			out << endl;
			for (int i = 0; i < lvl; ++i) out << "\t";
			l = 0;
		}
		out << word << " ";
		l += word.size() + 1;
	}
	out << endl;

}


void add_to_synopsis(ostream& out, Argument *arg, text type) {

	out << ez::essential::Terminal::bold( "\t--" + arg->get_long_label() );
	if (type.length() != 0) {
		out << "=" << ez::essential::Terminal::underline( type );
	}
	if (arg->get_short_label() != '\0') {
		out << " or " << ez::essential::Terminal::bold( "-" + arg->get_short_label() );
		if (type.length() != 0) {
			out << " " << ez::essential::Terminal::b_underline << type << ez::essential::Terminal::e_underline;
		}
	}

}


void ArgumentParser::print_synopsis(std::ostream& out) {

	out << eze::Terminal::bold( "NAME" ) << endl;
	ostringstream oss;
	oss << _program_name << " - " << _program_desc;
	print_synopsis_text( out, 1, oss.str(), 60 );
	out << endl << eze::Terminal::bold( "SYNOPSIS" ) << endl ;
	oss.str( "" );
	oss << _program_name << " [OPTIONS] ... " ;
	print_synopsis_text( out, 1, oss.str(), 60 );
	out << endl << eze::Terminal::bold( "DESCRIPTION" ) << endl << endl;

	sort( _arguments.begin(), _arguments.end(), [](Argument *a1, Argument *a2) {
		return a1->get_long_label() < a2->get_long_label();
	});
	
	for (auto arg : _arguments) {
		if (arg->hidden()) continue;

		text value_type = "";
		ostringstream message;
		if (typeid(*arg) == typeid(FlagArgument)) {
			add_to_synopsis(out, arg, "");
		} else if (typeid(*arg) == typeid(BooleanArgument)) {
			add_to_synopsis(out, arg, "BOOLEAN");
		} else if (typeid(*arg) == typeid(IntegerArgument)) {
			add_to_synopsis(out, arg, "INTEGER");
		} else if (typeid(*arg) == typeid(NaturalArgument)) {
			add_to_synopsis(out, arg, "NATURAL");
		} else if (typeid(*arg) == typeid(TextArgument)) {
			add_to_synopsis(out, arg, "text");
		} else if (typeid(*arg) == typeid(RealArgument)) {
			add_to_synopsis(out, arg, "FLOAT");
		} else if (typeid(*arg) == typeid(OptionsArgument)) {
			add_to_synopsis(out, arg, "VALUE");
			out << "\n\t  where value=" << dynamic_cast<OptionsArgument *>(arg)->get_allowed_options();
		} else if (typeid(*arg) == typeid(RealRangeArgument)) {
			add_to_synopsis(out, arg, "FLOAT");
			RealRangeArgument *ra = dynamic_cast<RealRangeArgument *>(arg);
			message << "where value is in [" << ra->min() << ".." << ra->max() << "]";
		} else if (typeid(*arg) == typeid(IntegerRangeArgument)) {
			add_to_synopsis(out, arg, "INTEGER");
			IntegerRangeArgument *ra = dynamic_cast<IntegerRangeArgument *>(arg);
			message << "where value is in [" << ra->min() << ".." << ra->max() << "]";
		} else if (typeid(*arg) == typeid(NaturalRangeArgument)) {
			add_to_synopsis(out, arg, "NATURAL");
			NaturalRangeArgument *ra = dynamic_cast<NaturalRangeArgument *>(arg);
			message << "where value is in [" << ra->min() << ".." << ra->max() << "]";
		}
		out << endl;
		text to_output = arg->get_description();
		if (message.str().length() > 0) {
			to_output += ", ";
			to_output += message.str();
		}
		
		print_synopsis_text(out, 2, to_output, 60);
		out << endl;
	}
	
}

void ArgumentParser::report_error(exception& e, bool show_synopsis) {
	if (_show_synopsis || show_synopsis) {
		print_synopsis(cout);
	}

	cerr << eze::Terminal::line_type_3 << endl;
	cerr << "Exception raised in program " << _program_name << endl;
	cerr << eze::Terminal::line_type_4 << endl;
	string msg = e.what();
	print_synopsis_text(cerr, 0, msg, 60);
	cerr << eze::Terminal::line_type_4 << endl;

	exit(EXIT_FAILURE);
}




