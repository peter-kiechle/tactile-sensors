#ifndef UTILS_H_
#define UTILS_H_

#include <cmath>
#include <algorithm>
#include <limits>       // std::numeric_limits
#include <string>

#include <stdint.h>

/**
 * @file
 * Just a few utilities functions.
 * Possibly copy-pasted from somewhere else...
 */

namespace utils
{

	/**
	 * Returns the current time in milliseconds, based on gettimeofday().
	 * @return Time in milliseconds.
	 */
	uint64_t getCurrentTimeMilliseconds();


	/**
	 * Floating point comparison, based on
     * @details Units in the Last Place: The larger the value, the larger the tolerance.
     *          ULP 0: x == y
     *          ULP 4: Suitable for 80 bits precision
	 * @param x The first float.
	 * @param y The second float.
	 * @param ulp Units in the Last Place.
	 * @return True if floats are almost equal.
	 */
	inline bool almostEqual(float x, float y, int ulp) {
		return abs(x-y) <= std::numeric_limits<float>::epsilon() * std::max(abs(x), abs(y))	* ulp;
	}


	/**
	 * Converts numbers to strings.
	 * @param number Some number.
	 * @return The string.
	 */
	template <typename T>
	std::string numberToString(T number);


	/**
	 * Converts string to number (if possible).
	 * @param text The string.
	 * @return The parsed number.
	 */
    template <typename T>
    T stringToNumber(const std::string &text);


	/**
	 * Splits specified filename into base name and extension
	 * @param filename The file name.
	 * @param basename The base name.
	 * @param extension The extension.
	 */
	void splitFilename(const std::string &filename, std::string &basename, std::string &extension);


	/**
	 * Converts Degrees to Radians
	 * @param d Degree.
	 * @return Angle in Radians
	 */
	inline double degToRad(double d) {
	    return d * M_PI/180.0;
	}

}

#endif /* UTILS_H_ */
