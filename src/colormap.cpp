#include "colormap.h"

Colormap::Colormap() {}

Colormap::~Colormap() {
	delete colorTable;
}

float Colormap::interpolate(float value, float f0, float f1, float x0, float x1) {
	return f0 + (((f1 - f0) / (x1 - x0)) * (value - x0));
}

RGB Colormap::hsl2rgb(HSL &hsl) {
	double r = hsl.l; // default to gray
	double g = hsl.l;
	double b = hsl.l;
	double h = hsl.h * 6.0;
	double v = (hsl.l <= 0.5) ? (hsl.l * (1.0 + hsl.s)) : (hsl.l + hsl.s - hsl.l * hsl.s);
	if (v > 0) {
		double m;
		double sv;
		int sextant;
		double fract, vsf, mid1, mid2;

		m = hsl.l + hsl.l - v;
		sv = (v - m) / v;

		sextant = (int) h;
		fract = h - sextant;
		vsf = v * sv * fract;
		mid1 = m + vsf;
		mid2 = v - vsf;
		switch (sextant) {
		case 0:
			r = v;
			g = mid1;
			b = m;
			break;
		case 1:
			r = mid2;
			g = v;
			b = m;
			break;
		case 2:
			r = m;
			g = v;
			b = mid1;
			break;
		case 3:
			r = m;
			g = mid2;
			b = v;
			break;
		case 4:
			r = mid1;
			g = m;
			b = v;
			break;
		case 5:
			r = v;
			g = m;
			b = mid2;
			break;
		}
	}
	return RGB( (float)r, (float)g, (float)b );
}


HSL Colormap::rgb2hsl(RGB &rgb) {
	double r = rgb.color[0];
	double g = rgb.color[1];
	double b = rgb.color[2];
	double v;
	double m;
	double vm;
	double r2, g2, b2;

	HSL hsl(0, 0, 0); // default to black

	v = max(r, g);
	v = max(v, b);
	m = min(r, g);
	m = min(m, b);
	hsl.l = (m + v) / 2.0;
	if (hsl.l <= 0.0) {
		return hsl;
	}
	vm = v - m;
	hsl.s = vm;
	if (hsl.s > 0.0) {
		hsl.s /= (hsl.l <= 0.5) ? (v + m) : (2.0 - v - m);
	} else {
		return hsl;
	}
	r2 = (v - r) / vm;
	g2 = (v - g) / vm;
	b2 = (v - b) / vm;
	if (r == v) {
		hsl.h = (g == m ? 5.0 + b2 : 1.0 - g2);
	} else if (g == v) {
		hsl.h = (b == m ? 1.0 + r2 : 3.0 - b2);
	} else {
		hsl.h = (r == m ? 3.0 + g2 : 5.0 - r2);
	}
	hsl.h /= 6.0;
	return hsl;
}

float Colormap::limitColorRange(float value, float low, float high) {
	if(value < low) return low;
	if(value > high) return high;
	else return value;
}


RGB Colormap::getColorFromColormap(vector<RGB> &colormap, float value, InterpMethod interpolationMethod) {
	int nColors = colormap.size();
	RGB rgb;
	int x0_idx = floor(value * (nColors-1)); // Index of "lower" specified color in colormap
	int x1_idx = ceil(value * (nColors-1)); // Index of "higher" specified color in colormap

	if(x0_idx == x1_idx) { // No interpolation necessary
		return colormap[x0_idx];
	}

	float x0 = static_cast<float>(x0_idx) / (nColors-1);
	float x1 = static_cast<float>(x1_idx) / (nColors-1);

	// HSL interpolation (gives "nicer" results than interpolation in RGB)
	if(interpolationMethod == HSL_INTERPOLATION) {
		HSL hsl0 = rgb2hsl(colormap[x0_idx]);
		HSL hsl1 = rgb2hsl(colormap[x1_idx]);
		HSL hsl_interp;

		float distCCW = hsl1.h - hsl0.h;
		float distCW = (hsl0.h+1.0) - hsl1.h;

		if(distCCW < distCW) { // counter clockwise distance is shorter
		hsl0.h = fmod(hsl0.h+1.0, 1.0);
		}

		float result = interpolate(value, hsl0.h, hsl1.h, x0, x1);
		hsl_interp.h = fmod(result, 1.0f);
		hsl_interp.s = interpolate(value, hsl0.s, hsl1.s, x0, x1);
		hsl_interp.l = interpolate(value, hsl0.l, hsl1.l, x0, x1);

		// Convert back to RGB
		rgb = hsl2rgb(hsl_interp);
	}

	// RGB interpolation
	if(interpolationMethod == RGB_INTERPOLATION) {
		for(int i = 0; i < 3; i++) {
			float f0 = colormap[x0_idx].color[i];
			float f1 = colormap[x1_idx].color[i];
			rgb.color[i] = interpolate(value, f0, f1, x0, x1);
		}
	}
	return rgb;
}


void Colormap::createColorTable(ColorGradient colorGradient, int nColors) {

	colorTable = new RGB[nColors];

	// Matlab's "Hot" colormap
	if(colorGradient == HOT) {
		for(int i = 0; i < nColors; i++) {
			float value = static_cast<float>(i) / static_cast<float>(nColors);
			colorTable[i].r = limitColorRange(3*value-0, 0.0, 1.0);
			colorTable[i].g = limitColorRange(3*value-1, 0.0, 1.0);
			colorTable[i].b = limitColorRange(3*value-2, 0.0, 1.0);
		}
	}

	// Matlab's "Jet" colormap
	if(colorGradient == JET) {
		for(int i = 0; i < nColors; i++) {
			float fourValue = 4.0f * static_cast<float>(i) / static_cast<float>(nColors);
			colorTable[i].r = limitColorRange( min(fourValue - 1.5, -fourValue + 4.5), 0.0, 1.0 );
			colorTable[i].g = limitColorRange( min(fourValue - 0.5, -fourValue + 3.5), 0.0, 1.0 );
			colorTable[i].b = limitColorRange( min(fourValue + 0.5, -fourValue + 2.5), 0.0, 1.0 );
		}
	}

	// Gnuplot's rgbformulae 7, 5, 15
	if(colorGradient == GNUPLOT) {
		for(int i = 0; i < nColors; i++) {
			float value = static_cast<float>(i) / static_cast<float>(nColors);
			colorTable[i].r = limitColorRange(sqrt(value), 0.0, 1.0);
			colorTable[i].g = limitColorRange(value*value*value, 0.0, 1.0);
			colorTable[i].b = limitColorRange(sin( (360*M_PI/180) * value), 0.0, 1.0);
		}
	}

	// Cycles through HSL-colorspace varying hue
	if(colorGradient == RAINBOW) {
		for(int i = 0; i < nColors; i++) {
			float hue = static_cast<float>(i) / static_cast<float>(nColors);
			HSL hsl(hue, 0.8, 0.5);
			colorTable[i] = hsl2rgb(hsl);
		}
	}

	// Rainbow color map with perceived constant luminance (at least roughly)
	// Taken from: Face-based Luminance Matching for Perceptual Colormap Generation
	if(colorGradient == RAINBOW_ISOLUMINANT) {
		vector<RGB> colormap;
		colormap.push_back(RGB(0.847, 0.057, 0.057)); // red
		colormap.push_back(RGB(0.527,0.527,0.000)); // yellow
		colormap.push_back(RGB(0.000,0.592,0.000)); // green
		colormap.push_back(RGB(0.000,0.559,0.559)); // cyan
		colormap.push_back(RGB(0.316,0.316,0.991)); // blue
		colormap.push_back(RGB(0.718,0.000,0.718)); // magenta

		for(int i = 0; i < nColors; i++) {
			float value = static_cast<float>(i) / static_cast<float>(nColors);
			colorTable[i] = getColorFromColormap(colormap, value, RGB_INTERPOLATION);
		}
	}

	// Red Yellow Blue
	// Based on www.ColorBrewer.org, by Cynthia A. Brewer, Penn State.
	// http://www.personal.psu.edu/cab38/ColorBrewer/ColorBrewer_RGB.html
	if(colorGradient == BREWER_SPECTRAL) {
		vector<RGB> colormap;
		colormap.push_back(RGB(0.61960784, 0.00392157, 0.25882353) );
		colormap.push_back(RGB(0.83529412, 0.24313725, 0.30980392) );
		colormap.push_back(RGB(0.95686275, 0.42745098, 0.26274510) );
		colormap.push_back(RGB(0.99215686, 0.68235294, 0.38039216) );
		colormap.push_back(RGB(0.99607843, 0.87843137, 0.54509804) );
		colormap.push_back(RGB(1.00000000, 1.00000000, 0.74901961) );
		colormap.push_back(RGB(0.90196078, 0.96078431, 0.59607843) );
		colormap.push_back(RGB(0.67058824, 0.86666667, 0.64313725) );
		colormap.push_back(RGB(0.40000000, 0.76078431, 0.64705882) );
		colormap.push_back(RGB(0.19607843, 0.53333333, 0.74117647) );
		colormap.push_back(RGB(0.36862745, 0.30980392, 0.63529412) );

		for(int i = 0; i < nColors; i++) {
			float value = 1.0 - static_cast<float>(i) / static_cast<float>(nColors);
			colorTable[i] = getColorFromColormap(colormap, value, RGB_INTERPOLATION);
		}
	}

	// Yellow Orange Red
	// Based on www.ColorBrewer.org, by Cynthia A. Brewer, Penn State.
	// http://www.personal.psu.edu/cab38/ColorBrewer/ColorBrewer_RGB.html
	if(colorGradient == BREWER_YlOrRd) {
		vector<RGB> colormap;
		colormap.push_back(RGB(1.00000000, 1.00000000, 0.80000000));
		colormap.push_back(RGB(1.00000000, 0.92941176, 0.62745098));
		colormap.push_back(RGB(0.99607843, 0.85098039, 0.46274510));
		colormap.push_back(RGB(0.99607843, 0.69803922, 0.29803922));
		colormap.push_back(RGB(0.99215686, 0.55294118, 0.23529412));
		colormap.push_back(RGB(0.98823529, 0.30588235, 0.16470588));
		colormap.push_back(RGB(0.89019608, 0.10196078, 0.10980392));
		colormap.push_back(RGB(0.74117647, 0.00000000, 0.14901961));
		colormap.push_back(RGB(0.50196078, 0.00000000, 0.14901961));
		colormap.push_back(RGB(0.3294117, 0.00000000, 0.09803921)); // Dark Red (does not belong to Brewer)

		for(int i = 0; i < nColors; i++) {
			float value = 1.0 - static_cast<float>(i) / static_cast<float>(nColors);
			colorTable[i] = getColorFromColormap(colormap, value, RGB_INTERPOLATION);
		}
	}

	// Yellow Green Blue
	// Based on www.ColorBrewer.org, by Cynthia A. Brewer, Penn State.
	// http://www.personal.psu.edu/cab38/ColorBrewer/ColorBrewer_RGB.html
	if(colorGradient == BREWER_YlGrBu) {
		vector<RGB> colormap;
		colormap.push_back(RGB(1.00000000, 1.00000000, 0.85098039));
		colormap.push_back(RGB(0.92941176, 0.97254902, 0.69411765));
		colormap.push_back(RGB(0.78039216, 0.91372549, 0.70588235));
		colormap.push_back(RGB(0.49803922, 0.80392157, 0.73333333));
		colormap.push_back(RGB(0.25490196, 0.71372549, 0.76862745));
		colormap.push_back(RGB(0.11372549, 0.56862745, 0.75294118));
		colormap.push_back(RGB(0.13333333, 0.36862745, 0.65882353));
		colormap.push_back(RGB(0.14509804, 0.20392157, 0.58039216));
		colormap.push_back(RGB(0.03137255, 0.11372549, 0.34509804));

		for(int i = 0; i < nColors; i++) {
			float value = 1.0 - static_cast<float>(i) / static_cast<float>(nColors);
			colorTable[i] = getColorFromColormap(colormap, value, RGB_INTERPOLATION);
		}
	}

	// Blue Yellow Red
	// DSA-Explorer
	if(colorGradient == DSA_EXPLORER) {
		vector<RGB> colormap;
		colormap.push_back(RGB(0.00000000, 0.00000000, 1.00000000));
		colormap.push_back(RGB(0.28235294, 0.28235294, 0.71764705));
		colormap.push_back(RGB(0.56862745, 0.56862745, 0.43137254));
		colormap.push_back(RGB(0.85098039, 0.85098039, 0.14901960));
		colormap.push_back(RGB(1.00000000, 0.85882352, 0.00000000));
		colormap.push_back(RGB(1.00000000, 0.57647058, 0.00000000));
		colormap.push_back(RGB(1.00000000, 0.29019607, 0.00000000));
		colormap.push_back(RGB(1.00000000, 0.00000000, 0.00000000));

		for(int i = 0; i < nColors; i++) {
			float value = static_cast<float>(i) / static_cast<float>(nColors);
			colorTable[i] = getColorFromColormap(colormap, value, RGB_INTERPOLATION);
		}
	}

	if(colorGradient == TWO_COLORS) {
		vector<RGB> colormap;

		//colormap.push_back(RGB(0.40392157, 0.0, 0.12156863)); // Red
		//colormap.push_back(RGB(0.0, 0.61568, 0.87843)); // Bright blue

		colormap.push_back(RGB(0.40392157, 0.0, 0.12156863)); // Red
		colormap.push_back(RGB(0.9, 0.9, 0.9)); // White

		for(int i = 0; i < nColors; i++) {
			float value = static_cast<float>(i) / static_cast<float>(nColors);
			colorTable[i] = getColorFromColormap(colormap, value, RGB_INTERPOLATION);
		}
	}

	// Oranges
	if(colorGradient == ORANGES) {
		vector<RGB> colormap;
		colormap.push_back(RGB(1.0, 0.5, 0.0));
		colormap.push_back(RGB(1.0, 0.5490196078431373, 0.098039215686274495));
		colormap.push_back(RGB(1.0, 0.59999999999999998, 0.19999999999999996));
		colormap.push_back(RGB(1.0, 0.64901960784313728, 0.29803921568627456));
		colormap.push_back(RGB(1.0, 0.69999999999999996, 0.40000000000000002));
		colormap.push_back(RGB(1.0, 0.74901960784313726, 0.49803921568627452));
		colormap.push_back(RGB(1.0, 0.80000000000000004, 0.59999999999999998));
		colormap.push_back(RGB(1.0, 0.85098039215686272, 0.70196078431372544));
		colormap.push_back(RGB(1.0, 0.90000000000000002, 0.80000000000000004));
		colormap.push_back(RGB(1.0, 0.9509803921568627, 0.90196078431372551));
		colormap.push_back(RGB(1.0, 1.0, 1.0));

		for(int i = 0; i < nColors; i++) {
			float value = static_cast<float>(i) / static_cast<float>(nColors);
			colorTable[i] = getColorFromColormap(colormap, value, RGB_INTERPOLATION);
		}
	}

	// Yellow Red
	if(colorGradient == YELLOW_RED) {
		vector<RGB> colormap;
		//colormap.push_back(RGB(0.2, 0.2, 0.2));
		colormap.push_back(RGB(0.8, 0.8, 0.0));
		colormap.push_back(RGB(0.8, 0.0, 0.0));

		for(int i = 0; i < nColors; i++) {
			float value = static_cast<float>(i) / static_cast<float>(nColors);
			colorTable[i] = getColorFromColormap(colormap, value, RGB_INTERPOLATION);
		}
	}

	// Yellow Green Red
	if(colorGradient == GREEN_YELLOW_RED) {
		vector<RGB> colormap;
		colormap.push_back(RGB(0.0, 1.0, 0.0));
		colormap.push_back(RGB(1.0, 1.0, 0.0));
		colormap.push_back(RGB(1.0, 0.0, 0.0));

		for(int i = 0; i < nColors; i++) {
			float value = static_cast<float>(i) / static_cast<float>(nColors);
			colorTable[i] = getColorFromColormap(colormap, value, RGB_INTERPOLATION);
		}
	}

	// Experimental
	if(colorGradient == EXPERIMENTAL) {
		vector<RGB> colormap;

		// Black & white
		colormap.push_back(RGB(0.35, 0.35, 0.35));
		colormap.push_back(RGB(1.0, 1.0, 1.0));

		int thresh_low = 31;
		int thresh_high = 47;
		for(int i = 0; i < nColors; i++) {
			if(i < thresh_low) {
				colorTable[i] = RGB(0.35, 0.35, 0.35);
			} else if(i > thresh_high) {
				colorTable[i] = RGB(1.0, 1.0, 1.0);
			}else {
				float value = static_cast<float>(i-thresh_low) / static_cast<float>(thresh_high-thresh_low);
				colorTable[i] = getColorFromColormap(colormap, value, RGB_INTERPOLATION);
			}
		}
	}
}


RGB& Colormap::getColorFromTable(int position) {
	return colorTable[position];
}
