#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <sstream>

#include "utils.h"

using namespace std;

namespace utils
{
	uint64_t getCurrentTimeMilliseconds() {
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
	}


	template <typename T>
	std::string numberToString(T number) {
		ostringstream ss;
		ss << number;
		return ss.str();
	}


	template <typename T>
	T stringToNumber(const std::string &text) {
		istringstream ss(text);
		T result;
		return ss >> result ? result : 0;
	}

	// Splits specified filename into base name and extension
	// Alternative:
	// #include "boost/filesystem.hpp"
	// boost::filesystem::path path("/home/user/foo.bar");
	// std::string basename = boost::filesystem::basename(path);
	// std::string extension = boost::filesystem::extension(path);
	void splitFilename(const std::string &filename, std::string &basename, std::string &extension) {
		string::size_type index = filename.find_last_of('.'); //search for last "." in file name

		if(index == string::npos) { // File name does not contain any period
			basename = filename;
			extension = "";
		} else { // Split at period
			basename = filename.substr(0, index);
			extension = filename.substr(index + 1);
		}
	}
}

