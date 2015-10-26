#ifndef COLORMAP_H_
#define COLORMAP_H_

#include <vector>
#include <cmath>

using namespace std;

/// Interpolation in RGB is faster but conversion to HSL results in better quality
enum InterpMethod {
	RGB_INTERPOLATION,
	HSL_INTERPOLATION
};

/// Some well known and experimental color maps
enum ColorGradient {
	HOT,
	JET,
	GNUPLOT,
	RAINBOW,
	RAINBOW_ISOLUMINANT,
	BREWER_SPECTRAL,
	BREWER_YlOrRd,
	BREWER_YlGrBu,
	DSA_EXPLORER,
	TWO_COLORS,
	ORANGES,
	YELLOW_RED,
	GREEN_YELLOW_RED,
	EXPERIMENTAL
};

/**
 * @class RGB
 * @brief Simple RGB color management class.
 *        Red Green Blue are floats in the range [0.0, 1.0].
 */
class RGB {
public:
	/**
	 * Constructor
	 * @param r,g,b Red, Green and Blue in the range [0.0, 1.0].
	 */
	RGB(float red = 0.0, float green = 0.0, float blue = 0.0) {
		r = red;
		g = green;
		b = blue;
	}
	/// anonymous structure: rgb.r = rgb.color[0]
	union {
		float color[3];
		struct {
			float r;
			float g;
			float b;
		};
	};
};

/**
 * @class HSL
 * @brief Simple HSL color management class.
 *        Hue, Saturation and Luminance are doubles in the range [0.0, 1.0].
 */
class HSL {
public:
	/**
	 * Constructor
	 * @param h,s,l Hue, saturation and luminance in the range [0.0, 1.0].
	 */
	HSL(double hue = 0.0, double saturation = 0.0, double luminance = 0.0) {
		h= hue;
		s = saturation;
		l = luminance;
	}
	/// anonymous structure: hsl.hue = hsl.data[0]
	union {
		double data[3];
		struct {
			double h;
			double s;
			double l;
		};
	};
};

/**
 * @class Colormap
 * @brief Manages colors and colormaps
 */
class Colormap {

public:
	Colormap();
	virtual ~Colormap();

	/**
	 * @brief Linear interpolation
	 * @details
	 *     .
	 *     |
	 *  f1 |-----------------*
	 *     |               . |
	 *     |            .    |
	 *   ? |---------X       |
	 *     |      .  |       |
	 *  f0 |---*     |       |
	 *     |.__|_____|_______|____.
	 *         x0   value    x1
	 *
	 *  @note \a x0 has to be smaller than \a x1
	 *  @param value Value between \a x0 and \a x1
	 *  @param f0,f1 Function values at supporting points.
 	 *  @param x0,x1 Supporting points.
	 *  @return Interpolated function value.
	 */
	float interpolate(float value, float f0, float f1, float x0, float x1);

	/**
	 * @brief Conversion from HSL to RGB color space
	 * @details Range: h[0..1], s[0..1], l[0..1],
	 *                 r[0..1], g[0..1], b[0..1]
	 * @param hsl Reference to HSL struct
	 * @return Resulting RGB struct.
	 */
	RGB hsl2rgb(HSL& hsl);

	/**
	 * @brief Conversion from RGB to HSL color space.
	 * @details Range: h[0..1], s[0..1], l[0..1],
	 *                 r[0..1], g[0..1], b[0..1]
	 * @param rgb Reference to RGB struct.
	 * @return Resulting HSL struct.
	 */
	HSL rgb2hsl(RGB &rgb);

	/**
	 * Clamps value to [low..high].
	 * @param value The value to be clamped.
	 * @param low The lower limit.
	 * @param high The upper limit.
	 * @return The clamped value.
	 */
	float limitColorRange(float value, float low, float high);

	/**
	 * Interpolates color between neighboring colors of a colormap.
	 * @param colormap The colormap, a vector of colors.
	 * @param value The desired color in the range [0.0, 1.0].
	 * @param interpolationMethod The interpolation method, HSL_INTERPOLATION or RGB_INTERPOLATION.
	 * @return The resulting color in RGB.
	 */
	RGB getColorFromColormap(vector<RGB> &colormap, float value, InterpMethod interpolationMethod = HSL_INTERPOLATION);

	/**
	 * Creates a predefined colormap.
	 * @param colorGradient Name (enum) of the actual colormap.
	 * @param nColors Number of Colors.
	 * @return void
	 */
	void createColorTable(ColorGradient colorGradient, int nColors);

	/**
	 * Returns entry of the colormap at specified position.
	 * @param position Position of the color in the colormap.
	 * @return A reference to the requested RGB struct.
	 */
	RGB& getColorFromTable(int position);

private:
	RGB *colorTable;
};

#endif /* COLORMAP_H_ */
