#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream> 
#include <iomanip> 
#include <boost/random.hpp>

#include <boost/thread.hpp>
#include <boost/chrono.hpp>

#include "guiSlipDetectionMultiPlot.h"

double const pi_div_180 = M_PI/180; // I know, this is probably already defined somewhere else...

NiceScale::NiceScale() { }

NiceScale::NiceScale(double min, double max, int maxTicks) {
	computeScale(min, max, maxTicks);
}

NiceScale::~NiceScale() { }

void NiceScale::computeScale(double min, double max, int maxTicks) {
	m_range = niceNum(max - min, false);
	m_tickSpacing = niceNum(m_range / (maxTicks - 1), true);
	m_niceMin = floor(min / m_tickSpacing) * m_tickSpacing;
	m_niceMax = ceil(max / m_tickSpacing) * m_tickSpacing;
	m_numTicks = static_cast<int>((m_niceMax-m_niceMin)/m_tickSpacing);
}

double NiceScale::niceNum(double range, bool round) {
	double exponent; // exponent of range
	double fraction; // fractional part of range
	double niceFraction; // nice, rounded fraction

	exponent = floor(log10(range));
	fraction = range / pow(10, exponent);

	if(round) {
		if (fraction < 1.5)
			niceFraction = 1;
		else if (fraction < 3)
			niceFraction = 2;
		else if (fraction < 7)
			niceFraction = 5;
		else
			niceFraction = 10;
	} else {
		if (fraction <= 1)
			niceFraction = 1;
		else if (fraction <= 2)
			niceFraction = 2;
		else if (fraction <= 5)
			niceFraction = 5;
		else
			niceFraction = 10;
	}

	return niceFraction * pow(10, exponent);
}

double NiceScale::getNiceMin() {
	return m_niceMin;
}

double NiceScale::getNiceMax() {
	return m_niceMax;
}

double NiceScale::getTickSpacing() {
	return m_tickSpacing;
}

int NiceScale::getNumTicks() {
	return m_numTicks;
}


// ----------------------------------------------------------------------------
// Current Slip Vector
// ----------------------------------------------------------------------------

SlipVectorLive::SlipVectorLive() {

	m_initialized = false;
	data_available = false;

	set_size_request(200, 200);
	set_double_buffered(true);

	#ifndef GLIBMM_DEFAULT_SIGNAL_HANDLERS_ENABLED
	//Connect the signal handler if it isn't already a virtual method override:
	signal_expose_event().connect(sigc::mem_fun(*this, &SlipVectorLive::on_expose_event), false);
	#endif //GLIBMM_DEFAULT_SIGNAL_HANDLERS_ENABLED
}


SlipVectorLive::~SlipVectorLive() { }


void SlipVectorLive::reset() {
	data_available = false;
	queue_draw();
}


bool SlipVectorLive::on_expose_event(GdkEventExpose* event) {

	Glib::RefPtr<Gdk::Window> window = get_window();

	if(window) {

		Gtk::Allocation allocation = get_allocation();
		const int width = allocation.get_width();
		const int height = allocation.get_height();

		// Create new cairo m_surface and context
		m_surface.clear();  // destroy previous m_surface (of possibly different size)
		m_surface = Cairo::ImageSurface::create(Cairo::FORMAT_ARGB32, width, height);
		Cairo::RefPtr<Cairo::Context> cr_surface = Cairo::Context::create(m_surface);

		// Draw empty graph on m_surface
		drawAxes(cr_surface, width, height);

		// Create context for the widget
		Cairo::RefPtr<Cairo::Context> cr = window->create_cairo_context();

		// Show created empty graph
		cr->set_source(m_surface, 0.0, 0.0);
		cr->paint();

		m_initialized = true;

		if(data_available) {
			drawVector(m_x, m_y);
		}
	}
	return true;
}


void SlipVectorLive::drawAxes(const Cairo::RefPtr<Cairo::Context>& cr, int width, int height) {

	m_axis_limit_x[0] = -3;
	m_axis_limit_x[1] = 3;
	m_axis_limit_y[0] = -3;
	m_axis_limit_y[1] = 3;
	m_axis_range_x = m_axis_limit_x[1]-m_axis_limit_x[0];
	m_axis_range_y = m_axis_limit_y[1]-m_axis_limit_y[0];

	m_padding_left = 0.2*m_axis_range_x;
	m_padding_right = 0.05*m_axis_range_x;
	m_padding_bottom = 0.2*m_axis_range_y;
	m_padding_top = 0.05*m_axis_range_y;

	m_scale_x = width/(m_axis_range_x+m_padding_left+m_padding_right);
	m_scale_y = height/(m_axis_range_y+m_padding_top+m_padding_bottom);
	m_offset_x = m_padding_left + 0.5*(m_axis_range_x);
	m_offset_y = m_padding_top + 0.5*(m_axis_range_y);

	// Scale to workspace dimension and translate (0, 0) to be the center
	cr->scale(m_scale_x, m_scale_y);
	cr->translate(m_offset_x, m_offset_y);

	// Background
	cr->set_source_rgba(0.85, 0.85, 0.85, 1.0);
	cr->paint();

	//----------
	// Axis box
	//----------
	cr->save();
	cr->set_source_rgba(1.0, 1.0, 1.0, 1.0);
	cr->rectangle(m_axis_limit_x[0], m_axis_limit_y[0], m_axis_range_x, m_axis_range_y);
	cr->fill_preserve(); // background
	cr->set_line_width(0.03);
	cr->set_line_cap(Cairo::LINE_CAP_SQUARE);
	cr->set_source_rgba(0.3, 0.3, 0.3, 1.0);
	cr->stroke(); // outline
	cr->restore();

	//------------
	// Grid lines
	//------------
	cr->save();
	cr->set_line_width(0.03);
	cr->set_line_cap(Cairo::LINE_CAP_SQUARE);
	cr->set_source_rgba(0.3, 0.3, 0.3, 0.3);
	std::vector< double > dashes(2);
	dashes[0] = 0.05;
	dashes[1] = 0.05;
	cr->set_dash(dashes, 0.0);
	// vertical
	for(int i = 1; i < m_axis_range_x; i++) {
		cr->move_to(m_axis_limit_x[0] + i, m_axis_limit_y[1]);
		cr->line_to(m_axis_limit_x[0] + i, m_axis_limit_y[0]);
	}
	// horizontal
	for(int i = 1; i < m_axis_range_x; i++) {
		cr->move_to(m_axis_limit_x[0], m_axis_limit_y[0] + i);
		cr->line_to(m_axis_limit_x[1], m_axis_limit_y[0] + i);
	}
	cr->stroke();
	cr->restore();

	//------------
	// Axis ticks
	//------------
	cr->save();
	cr->set_line_width(0.03);
	cr->set_line_cap(Cairo::LINE_CAP_SQUARE);
	cr->set_source_rgba(0.3, 0.3, 0.3, 1.0);
	// xticks
	double tick_length = 0.2;
	for(int i = 1; i < m_axis_range_x; i++) {
		// bottom
		cr->move_to(m_axis_limit_x[0] + i, m_axis_limit_y[1]);
		cr->line_to(m_axis_limit_x[0] + i, m_axis_limit_y[1] + 0.5*tick_length);
		// top
		cr->move_to(m_axis_limit_x[0] + i, m_axis_limit_y[0]);
		cr->line_to(m_axis_limit_x[0] + i, m_axis_limit_y[0] + 0.5*tick_length);
	}
	// yticks
	for(int i = 1; i < m_axis_range_x; i++) {
		// left
		cr->move_to(m_axis_limit_x[0], m_axis_limit_y[0] + i);
		cr->line_to(m_axis_limit_x[0] + 0.5*tick_length, m_axis_limit_y[0] + i);
		// right
		cr->move_to(m_axis_limit_x[1], m_axis_limit_y[0] + i);
		cr->line_to(m_axis_limit_x[1] - 0.5*tick_length, m_axis_limit_y[0] + i);
	}
	cr->stroke();
	cr->restore();

	//-------------
	// Axis labels
	//-------------
	cr->save();
	Cairo::RefPtr<Cairo::ToyFontFace> font = Cairo::ToyFontFace::create("sans", Cairo::FONT_SLANT_NORMAL, Cairo::FONT_WEIGHT_NORMAL);
	cr->set_font_face(font);
	cr->set_font_size(0.3);
	cr->set_source_rgba(0.0, 0.0, 0.0, 1.0);
	std::ostringstream ss;
	std::string valueStr;
	Cairo::TextExtents te;

	// x-axis
	valueStr = "Δx";
	cr->get_text_extents(valueStr, te);
	double offsetX = 0.5 * te.width;
	double offsetY = 0.5 * te.height;
	cr->move_to(-offsetX, m_axis_limit_y[1] + 0.8*m_padding_bottom + offsetY);

	cr->show_text(valueStr);
	for(int i = 0; i < m_axis_range_x+1; i++) {
		ss.str(""); // clear
		ss << m_axis_limit_x[0] + i;
		valueStr = ss.str();
		cr->get_text_extents(valueStr, te);
		offsetX = 0.5 * te.width;
		offsetY = 0.5 * te.height;
		cr->move_to(m_axis_limit_x[0] + i - offsetX, m_axis_limit_y[1] + 0.4*m_padding_bottom + offsetY);
		cr->show_text(valueStr);
	}

	// y-axis
	valueStr = "Δy";
	cr->get_text_extents(valueStr, te);
	offsetX = 0.5 * te.width;
	offsetY = 0.5 * te.height;
	cr->move_to(m_axis_limit_x[0] - 0.8*m_padding_left - offsetX, offsetY);
	cr->show_text(valueStr);
	for(int i = 0; i < m_axis_range_y+1; i++) {
		ss.str(""); // clear
		ss << m_axis_limit_y[0] + i;
		valueStr = ss.str();
		cr->get_text_extents(valueStr, te);
		offsetX = 0.5 * te.width;
		offsetY = 0.5 * te.height;
		cr->move_to(m_axis_limit_x[0] - 0.4*m_padding_left - offsetX, m_axis_limit_y[0] + i + offsetY);
		cr->show_text(valueStr);
		ss.str(""); // clear
	}
	cr->restore();
}


bool SlipVectorLive::drawVector(double x, double y) {
	Glib::RefPtr<Gdk::Window> window = get_window();

	if(window && m_initialized) {

		m_x = x;
		m_y = y;
		data_available = true;

		// Create cairo context for the widget
		Cairo::RefPtr<Cairo::Context> cr = window->create_cairo_context();

		// Draw empty graph
		cr->set_source(m_surface, 0.0, 0.0);
		cr->paint();

		// Scale to workspace dimension and translate (0, 0) to be the center
		cr->scale(m_scale_x, m_scale_y);
		cr->translate(m_offset_x, m_offset_y);

		// Circle
		double radius = sqrt(x*x + y*y);
		cr->arc(0.0, 0.0, radius, 0, 2*M_PI);
		//cr->set_source_rgba(0.1, 0.1, 0.1, 1.0); // black
		//cr->stroke_preserve();
		cr->set_source_rgba(0.0, 0.1765, 0.4392, 0.2); // uibk blue
		//cr->set_source_rgba(1.0, 0.5, 0.0, 1.0); // uibk orange
		cr->fill();

		// Vector
		cr->set_source_rgba(0.0, 0.1765, 0.4392, 1.0); // uibk blue
		cr->set_line_width(0.05);
		cr->set_line_cap(Cairo::LINE_CAP_BUTT);

		cr->move_to(0, 0);
		cr->line_to(x,y);
		cr->stroke();
	}
	return true;
}


// ----------------------------------------------------------------------------
// Slip vector trajectory over time
// ----------------------------------------------------------------------------

SlipVectorTrajectory::SlipVectorTrajectory(std::deque<slip_trajectory>& slipvectors)
: m_slipvectors(slipvectors)
{

	m_initialized = false;
	data_available = false;

	set_size_request(200, 200);
	set_double_buffered(true);

	#ifndef GLIBMM_DEFAULT_SIGNAL_HANDLERS_ENABLED
	//Connect the signal handler if it isn't already a virtual method override:
	signal_expose_event().connect(sigc::mem_fun(*this, &guiSlipDetectionMultiPlot::on_expose_event), false);
	#endif //GLIBMM_DEFAULT_SIGNAL_HANDLERS_ENABLED
}

SlipVectorTrajectory::~SlipVectorTrajectory() { }


void SlipVectorTrajectory::reset() {
	data_available = false;
	queue_draw();
}


bool SlipVectorTrajectory::on_expose_event(GdkEventExpose* event) {
	drawBackgroundSurface();
	m_initialized = true;
	if(data_available) {
		drawTrajectory(m_slipvectors, m_currentFrameID); // Draw entire trajectory
	}
	return true;
}

void SlipVectorTrajectory::drawBackgroundSurface() {
	Glib::RefPtr<Gdk::Window> window = get_window();

	if(window) {
		Gtk::Allocation allocation = get_allocation();
		const int width = allocation.get_width();
		const int height = allocation.get_height();

		// Create new cairo m_surface and context
		m_surface.clear();  // destroy previous m_surface (of possibly different size)
		m_surface = Cairo::ImageSurface::create(Cairo::FORMAT_ARGB32, width, height);
		Cairo::RefPtr<Cairo::Context> cr_surface = Cairo::Context::create(m_surface);

		// Draw empty graph on m_surface
		drawAxes(cr_surface, width, height);

		// Create context for the widget
		Cairo::RefPtr<Cairo::Context> cr = window->create_cairo_context();

		// Show created empty graph
		cr->set_source(m_surface, 0.0, 0.0);
		cr->paint();

	}
}


void SlipVectorTrajectory::drawAxes(const Cairo::RefPtr<Cairo::Context>& cr, int width, int height) {
	m_axis_limit_x[0] = -6;
	m_axis_limit_x[1] = 6;
	m_axis_limit_y[0] = -14;
	m_axis_limit_y[1] = 14;
	m_axis_range_x = m_axis_limit_x[1]-m_axis_limit_x[0];
	m_axis_range_y = m_axis_limit_y[1]-m_axis_limit_y[0];

	m_padding_left = 0.25*m_axis_range_x;
	m_padding_right = 0.05*m_axis_range_x;
	m_padding_top = 0.05*m_axis_range_y;
	m_padding_bottom = 0.15*m_axis_range_y;
	m_horizontal_spacer = m_axis_range_y+m_padding_top+m_padding_bottom - (m_axis_range_x+m_padding_left+m_padding_right);

	m_scale_x = width/(m_axis_range_x+m_padding_left+m_padding_right+m_horizontal_spacer);
	m_scale_y = height/(m_axis_range_y+m_padding_top+m_padding_bottom);
	m_offset_x = m_padding_left + 0.5*(m_axis_range_x+m_horizontal_spacer);
	m_offset_y =  m_padding_top + 0.5*(m_axis_range_y);

	// Scale to workspace dimension and translate (0, 0) to be the center
	cr->scale(m_scale_x, m_scale_y);
	cr->translate(m_offset_x, m_offset_y);

	// Background
	cr->set_source_rgba(0.85, 0.85, 0.85, 1.0);
	cr->paint();

	//----------
	// Axis box
	//----------
	cr->save();
	cr->set_source_rgba(1.0, 1.0, 1.0, 1.0);
	cr->rectangle(m_axis_limit_x[0], m_axis_limit_y[0], m_axis_range_x, m_axis_range_y);
	cr->fill_preserve(); // background
	cr->set_line_width(0.10);
	cr->set_line_cap(Cairo::LINE_CAP_SQUARE);
	cr->set_source_rgba(0.3, 0.3, 0.3, 1.0);
	cr->stroke(); // outline
	cr->restore();

	//------------
	// Grid lines
	//------------
	cr->save();
	cr->set_line_width(0.08);
	cr->set_line_cap(Cairo::LINE_CAP_SQUARE);
	std::vector< double > dashes(2);
	dashes[0] = 0.4;
	dashes[1] = 0.3;
	cr->set_dash(dashes, 0.0);
	cr->set_source_rgba(0.3, 0.3, 0.3, 0.3);
	// vertical lines
	for(int i = 0; i < m_axis_range_x+1; i+=2) {
		cr->move_to(m_axis_limit_x[0] + i, m_axis_limit_y[1]);
		cr->line_to(m_axis_limit_x[0] + i, m_axis_limit_y[0]);
	}
	// horizontal lines
	for(int i = 0; i < m_axis_range_y+1; i+=2) {
		cr->move_to(m_axis_limit_x[0], m_axis_limit_y[0] + i);
		cr->line_to(m_axis_limit_x[1], m_axis_limit_y[0] + i);
	}
	cr->stroke();
	cr->restore();

	//------------
	// Axis ticks
	//------------
	cr->save();
	cr->set_line_width(0.10);
	cr->set_line_cap(Cairo::LINE_CAP_SQUARE);
	cr->set_source_rgba(0.3, 0.3, 0.3, 1.0);
	// xticks
	double tick_length = 0.3;
	for(int i = 2; i < m_axis_range_x; i+=2) {
		// bottom
		cr->move_to(m_axis_limit_x[0] + i, m_axis_limit_y[1]);
		cr->line_to(m_axis_limit_x[0] + i, m_axis_limit_y[1] - 0.5*tick_length);
		// top
		cr->move_to(m_axis_limit_x[0] + i, m_axis_limit_y[0]);
		cr->line_to(m_axis_limit_x[0] + i, m_axis_limit_y[0] + 0.5*tick_length);
	}
	// yticks
	for(int i = 2; i < m_axis_range_y; i+=2) {
		// left
		cr->move_to(m_axis_limit_x[0], m_axis_limit_y[0] + i);
		cr->line_to(m_axis_limit_x[0] + 0.5*tick_length, m_axis_limit_y[0] + i);
		// right
		cr->move_to(m_axis_limit_x[1], m_axis_limit_y[0] + i);
		cr->line_to(m_axis_limit_x[1] - 0.5*tick_length, m_axis_limit_y[0] + i);
	}
	cr->stroke();
	cr->restore();

	//-------------
	// Axis labels
	//-------------
	cr->save();
	Cairo::RefPtr<Cairo::ToyFontFace> font = Cairo::ToyFontFace::create("sans", Cairo::FONT_SLANT_NORMAL, Cairo::FONT_WEIGHT_NORMAL);
	cr->set_font_face(font);
	cr->set_font_size(1.0);
	std::ostringstream ss;
	std::string valueStr;
	Cairo::TextExtents te;

	// x-axis
	valueStr = "Δx";
	cr->get_text_extents(valueStr, te);
	double offsetX = 0.5 * te.width;
	double offsetY = 0.5 * te.height;
	cr->move_to(-offsetX, m_axis_limit_y[1] + 0.8*m_padding_bottom + offsetY);
	cr->set_source_rgba(0.0, 0.0, 0.0, 1.0);
	cr->show_text(valueStr);
	for(int i = 0; i < m_axis_range_x+1; i+=2) {
		ss.str(""); // clear
		ss << m_axis_limit_x[0] + i;
		valueStr = ss.str();
		cr->get_text_extents(valueStr, te);
		offsetX = 0.5 * te.width;
		offsetY = 0.5 * te.height;
		cr->move_to(m_axis_limit_x[0] + i - offsetX, m_axis_limit_y[1] + 0.4*m_padding_bottom + offsetY);
		cr->set_source_rgb(0.0, 0.0, 0.0);
		cr->show_text(valueStr);
	}

	// y-axis
	valueStr = "Δy";
	cr->get_text_extents(valueStr, te);
	offsetX = 0.5 * te.width;
	offsetY = 0.5 * te.height;
	cr->move_to(m_axis_limit_x[0] - 1.0*m_padding_left - offsetX, offsetY);
	cr->set_source_rgba(0.0, 0.0, 0.0, 1.0);
	cr->show_text(valueStr);
	for(int i = 0; i < m_axis_range_y+1; i+=2) {
		ss.str(""); // clear
		ss << m_axis_limit_y[0] + i;
		valueStr = ss.str();
		cr->get_text_extents(valueStr, te);
		offsetX = 0.5 * te.width;
		offsetY = 0.5 * te.height;
		cr->move_to(m_axis_limit_x[0] - 0.5*m_padding_left - offsetX, m_axis_limit_y[0] + i + offsetY);
		cr->set_source_rgb(0.0, 0.0, 0.0);
		cr->show_text(valueStr);
		ss.str(""); // clear
	}
	cr->restore();
}


bool SlipVectorTrajectory::updateTrajectory(std::deque<slip_trajectory>& slipvectors, uint currentFrameID) {

	Glib::RefPtr<Gdk::Window> window = get_window();

	if(window && m_initialized) {

		m_slipvectors = slipvectors;
		m_currentFrameID = currentFrameID;
		data_available = true;

		// Create cairo context for the widget
		Cairo::RefPtr<Cairo::Context> cr = window->create_cairo_context();

		// Scale to workspace dimension and translate (0, 0) to be the center
		cr->scale(m_scale_x, m_scale_y);
		cr->translate(m_offset_x, m_offset_y);

		//----------------------------------------
		// Last segment of slip vector trajectory
		//----------------------------------------
		cr->set_source_rgba(0.0, 0.1765, 0.4392, 1.0); // uibk blue
		//cr->set_source_rgba(1.0, 0.5, 0.0, 1.0); // uibk orange
		cr->set_line_width(0.15);
		cr->set_line_cap(Cairo::LINE_CAP_ROUND);

		double x1, y1, x2, y2;
		uint n = currentFrameID; //slipvectors.size()-1;
		if(n == 0) {
			x1 = 0.0;
			x2 = 0.0;
		} else {
			boost::tie(x1, y1) = slipvectors[n-1];
		}
		boost::tie(x2, y2) = slipvectors[n];
		cr->move_to(x1, y1);
		cr->line_to(x2, y2);
		cr->stroke();
	}
	return true;

}


bool SlipVectorTrajectory::drawTrajectory(std::deque<slip_trajectory>& slipvectors, uint currentFrameID) {

	Glib::RefPtr<Gdk::Window> window = get_window();

	if(window && m_initialized) {

		m_slipvectors = slipvectors;
		m_currentFrameID = currentFrameID;
		data_available = true;

		// Create cairo context for the widget
		Cairo::RefPtr<Cairo::Context> cr = window->create_cairo_context();

		// Draw empty graph
		cr->set_source(m_surface, 0.0, 0.0);
		cr->paint();

		// Scale to workspace dimension and translate (0, 0) to be the center
		cr->scale(m_scale_x, m_scale_y);
		cr->translate(m_offset_x, m_offset_y);

		//------------------------
		// Slip vector trajectory
		//------------------------
		cr->set_source_rgba(0.0, 0.1765, 0.4392, 1.0); // uibk blue
		//cr->set_source_rgba(1.0, 0.5, 0.0, 1.0); // uibk orange
		cr->set_line_width(0.15);
		cr->set_line_cap(Cairo::LINE_CAP_ROUND);
		cr->move_to(0, 0);

		double x, y;
		for(uint frameID = 0; frameID <= currentFrameID; frameID++) {
			boost::tie(x, y) = slipvectors[frameID];
			cr->line_to(x, y);
		}
		cr->stroke();
	}
	return true;
}


// ----------------------------------------------------------------------------
//    Orientation
// ----------------------------------------------------------------------------

Orientation::Orientation() {

	m_initialized = false;
	data_available = false;

	set_size_request(200, 200);
	set_double_buffered(true);

	#ifndef GLIBMM_DEFAULT_SIGNAL_HANDLERS_ENABLED
	//Connect the signal handler if it isn't already a virtual method override:
	signal_expose_event().connect(sigc::mem_fun(*this, &SlipVectorLive::on_expose_event), false);
	#endif //GLIBMM_DEFAULT_SIGNAL_HANDLERS_ENABLED
}

Orientation::~Orientation() { }

void Orientation::reset() {
	data_available = false;
	queue_draw();
}


bool Orientation::on_expose_event(GdkEventExpose* event) {

	Glib::RefPtr<Gdk::Window> window = get_window();

	if(window) {

		Gtk::Allocation allocation = get_allocation();
		const int width = allocation.get_width();
		const int height = allocation.get_height();

		// Create new cairo m_surface and context
		m_surface.clear();  // destroy previous m_surface (of possibly different size)
		m_surface = Cairo::ImageSurface::create(Cairo::FORMAT_ARGB32, width, height);
		Cairo::RefPtr<Cairo::Context> cr_m_surface = Cairo::Context::create(m_surface);

		// Draw empty graph on m_surface
		drawAxes(cr_m_surface, width, height);

		// Create context for the widget
		Cairo::RefPtr<Cairo::Context> cr = window->create_cairo_context();

		// Show created empty graph
		cr->set_source(m_surface, 0.0, 0.0);
		cr->paint();

		m_initialized = true;
		if(data_available) {
			drawOrientation(m_success, m_angle, m_lambda1, m_lambda2, m_skew_x, m_skew_y);
		}
	}

	return true;
}


void Orientation::drawAxes(const Cairo::RefPtr<Cairo::Context>& cr, int width, int height) {

	// unit circle
	m_axis_limit_x[0] = -1;
	m_axis_limit_x[1] = 1;
	m_axis_limit_y[0] = -1;
	m_axis_limit_y[1] = 1;
	m_axis_range_x = m_axis_limit_x[1]-m_axis_limit_x[0];
	m_axis_range_y = m_axis_limit_y[1]-m_axis_limit_y[0];

	m_padding_left = 0.15*m_axis_range_x;
	m_padding_right = 0.30*m_axis_range_x;
	m_padding_bottom = 0.30*m_axis_range_y;
	m_padding_top = 0.15*m_axis_range_y;

	m_scale_x = width/(m_axis_range_x+m_padding_left+m_padding_right);
	m_scale_y = height/(m_axis_range_y+m_padding_top+m_padding_bottom);
	m_offset_x = m_padding_left + 0.5*(m_axis_range_x);
	m_offset_y = m_padding_top + 0.5*(m_axis_range_y);

	m_skew_x_pos = m_axis_limit_y[1]+0.20*m_axis_range_y;
	m_skew_y_pos = m_axis_limit_x[1]+0.20*m_axis_range_x;
	m_max_skew_x = 2.0;
	m_max_skew_y = 2.0;

	// Scale to workspace dimension and translate (0, 0) to be the center
	cr->scale(m_scale_x, m_scale_y);
	cr->translate(m_offset_x, m_offset_y);

	// Background
	cr->set_source_rgba(0.85, 0.85, 0.85, 1.0);
	cr->paint();

	//----------
	// Axis box (circle)
	//----------
	cr->save();
	cr->set_source_rgba(1.0, 1.0, 1.0, 1.0);
	cr->arc(0.0, 0.0, 1.0, 0, 2*M_PI);
	cr->fill_preserve(); // background
	cr->set_line_width(0.01);
	cr->set_line_cap(Cairo::LINE_CAP_SQUARE);
	cr->set_source_rgba(0.3, 0.3, 0.3, 1.0);
	cr->stroke(); // outline
	cr->restore();

	//------------
	// Grid lines
	//------------
	cr->save();
	cr->set_line_width(0.01);
	cr->set_line_cap(Cairo::LINE_CAP_BUTT);
	std::vector< double > dashes(2);
	dashes[0] = 0.02;
	dashes[1] = 0.04;
	cr->set_dash(dashes, 0.0);
	cr->set_source_rgba(0.3, 0.3, 0.3, 0.3);
	double dx, dy;
	for(double i = 0.0; i < 180.0; i+=30.0) {
		dx = cos(pi_div_180 * i);
		dy = sin(pi_div_180 * i);
		cr->move_to(-dx, -dy);
		cr->line_to( dx,  dy);
	}
	cr->stroke(); 
	cr->restore();


	//-------------
	// Axis labels
	//-------------
	cr->save();
	Cairo::RefPtr<Cairo::ToyFontFace> font = Cairo::ToyFontFace::create("sans", Cairo::FONT_SLANT_NORMAL, Cairo::FONT_WEIGHT_NORMAL);
	cr->set_font_face(font);
	cr->set_font_size(0.09);
	cr->set_source_rgba(0.0, 0.0, 0.0, 1.0);
	std::ostringstream ss;
	std::string valueStr;
	Cairo::TextExtents te;
	double offsetX, offsetY;
	for(double i = 0.0; i < 360.0; i+=30.0) {
		ss.str(""); // clear
		ss << i << "°";
		valueStr = ss.str();
		cr->get_text_extents(valueStr, te);
		offsetX = 0.5 * te.width;
		offsetY = 0.5 * te.height;
		dx = 1.15 * cos(pi_div_180 * ((360-i)+180.0) );
		dy = 1.15 * sin(pi_div_180 * ((360-i)+180.0) );
		dy = -dy; // invert y-axis
		cr->move_to(dx-offsetX, dy+offsetY);
		cr->show_text(valueStr);
	}
	cr->restore();

	//------------
	// Skew
	//------------
	cr->save();
	cr->set_line_width(0.02);
	cr->set_line_cap(Cairo::LINE_CAP_BUTT);
	std::vector< double > dashes_skew(2);
	dashes_skew[0] = 0.03;
	dashes_skew[1] = 0.03;
	cr->set_dash(dashes_skew, 0.0);
	cr->set_source_rgba(0.7, 0.7, 0.7, 1.0);
	// Dashed line x
	cr->move_to(-1.0, m_skew_x_pos);
	cr->line_to( 1.0, m_skew_x_pos);
	// Dashed line y
	cr->move_to(m_skew_y_pos, -1.0);
	cr->line_to(m_skew_y_pos,  1.0);
	cr->stroke();

	// Marker x
	cr->set_source_rgba(0.6, 0.6, 0.6, 1.0);
	cr->begin_new_sub_path();
	cr->arc(0.0, m_skew_y_pos, 0.05, 0, 2*M_PI);
	cr->fill();
	// Marker y
	cr->begin_new_sub_path();
	cr->arc(m_skew_x_pos, 0.0, 0.05, 0, 2*M_PI);
	cr->fill();

	cr->restore();
}


bool Orientation::drawOrientation(bool success, double angle, double lambda1, double lambda2, double skew_x, double skew_y) {

	Glib::RefPtr<Gdk::Window> window = get_window();

	if(window && m_initialized) {

		m_success = success;
		m_angle = angle;
		m_lambda1 = lambda1;
		m_lambda2 = lambda2;
		m_skew_x = skew_x;
		m_skew_y = skew_y;
		data_available = true;

		// Create cairo context for the widget
		Cairo::RefPtr<Cairo::Context> cr = window->create_cairo_context();

		// Draw empty graph
		cr->set_source(m_surface, 0.0, 0.0);
		cr->paint();

		// Scale to workspace dimension and translate (0, 0) to be the center
		cr->scale(m_scale_x, m_scale_y);
		cr->translate(m_offset_x, m_offset_y);


		if(success) {

			//--------------------------------------
			// Principal axes of Eigenvalue problem
			//--------------------------------------
			//cr->set_source_rgba(0.0, 0.1765, 0.4392, 1.0); // uibk blue
			cr->set_source_rgba(1.0, 0.5, 0.0, 1.0); // uibk orange
			cr->set_line_width(0.03);
			cr->set_line_cap(Cairo::LINE_CAP_BUTT);

			double major_axis_width = lambda1;
			double minor_axis_width = lambda2;

			// Scale principal axes
			double scale_factor = 1.0 / major_axis_width;
			major_axis_width *= scale_factor;
			minor_axis_width *= scale_factor;

			// major axis
			double dx = major_axis_width * cos(pi_div_180 * (180.0 - fmod(angle, 180.0)) );
			double dy = major_axis_width * sin(pi_div_180 * (180.0 - fmod(angle, 180.0)) );
			dy = -dy; // invert y-axis
			cr->move_to(-dx, -dy);
			cr->line_to( dx,  dy);

			// minor axis
			dx = minor_axis_width * cos( pi_div_180 * (180.0 - fmod(angle+90.0, 180.0)) );
			dy = minor_axis_width * sin( pi_div_180 * (180.0 - fmod(angle+90.0, 180.0)) );
			dy = -dy; // invert y-axis
			cr->move_to(-dx, -dy);
			cr->line_to( dx,  dy);
			cr->stroke();

			// Label current orientation
			Cairo::RefPtr<Cairo::ToyFontFace> font = Cairo::ToyFontFace::create("sans", Cairo::FONT_SLANT_NORMAL, Cairo::FONT_WEIGHT_BOLD);
			cr->set_font_face(font);
			cr->set_font_size(0.14);
			std::ostringstream ss;
			std::string valueStr;
			ss << std::fixed << std::setprecision(1) << angle << "°";
			valueStr = ss.str();
			Cairo::TextExtents te;
			cr->get_text_extents(valueStr, te);
			double offsetX = 0.5 * te.width;
			double offsetY = 0.5 * te.height;
			cr->move_to(-1.0-offsetX, -1.1-offsetY);
			cr->show_text(valueStr);

		} else {

			//--------------
			// Gray axis box
			//--------------
			cr->set_source_rgba(0.7, 0.7, 0.7, 1.0);
			cr->arc(0.0, 0.0, 1.0, 0, 2*M_PI);
			cr->fill();
		}

		//------------
		// Skew
		//------------

		// Skew x
		cr->set_source_rgba(1.0, 0.5, 0.0, 1.0); // uibk orange
		cr->begin_new_sub_path();
		cr->arc(-skew_x/m_max_skew_x, m_skew_y_pos, 0.05, 0, 2*M_PI);
		cr->fill_preserve();
		cr->set_line_width(0.01);
		cr->set_line_cap(Cairo::LINE_CAP_SQUARE);
		cr->set_source_rgba(0.3, 0.3, 0.3, 1.0);
		cr->stroke(); 

		// Skew y
		cr->set_source_rgba(1.0, 0.5, 0.0, 1.0); // uibk orange
		cr->begin_new_sub_path();
		cr->arc(m_skew_x_pos, -skew_y/m_max_skew_y, 0.05, 0, 2*M_PI);
		cr->fill_preserve();
		cr->set_line_width(0.01);
		cr->set_line_cap(Cairo::LINE_CAP_SQUARE);
		cr->set_source_rgba(0.3, 0.3, 0.3, 1.0);
		cr->stroke(); 

	}
	return true;
}



// ----------------------------------------------------------------------------
// Orientation trajectory over time
// ----------------------------------------------------------------------------
// Note:
// In order to avoid the transformation of datapoints, paths are created in a conveniently distorted scale context.
// For uniform line width, drawing operations take place in a 1:1 scale context

OrientationTrajectory::OrientationTrajectory(std::deque<double>& slipangles)
	: m_slipangles(slipangles)
{
	m_initialized = false;
	data_available = false;
	resetAxisLimits(0, 100, -100, 100);
	set_size_request(200, 200);
	set_double_buffered(true);

	#ifndef GLIBMM_DEFAULT_SIGNAL_HANDLERS_ENABLED
	//Connect the signal handler if it isn't already a virtual method override:
	signal_expose_event().connect(sigc::mem_fun(*this, &guiSlipDetectionMultiPlot::on_expose_event), false);
	#endif //GLIBMM_DEFAULT_SIGNAL_HANDLERS_ENABLED
}


OrientationTrajectory::~OrientationTrajectory() { }


void OrientationTrajectory::reset() {
	data_available = false;
	queue_draw();
}


void OrientationTrajectory::resetAxisLimits(int x_lower, int x_upper, int y_lower,  int y_upper) {
	m_axis_limit_x[0] = x_lower;
	m_axis_limit_x[1] = x_upper;
	m_axis_limit_y[0] = y_lower; // Assumption: < 0
	m_axis_limit_y[1] = y_upper; // Assumption: > 0
}


bool OrientationTrajectory::on_expose_event(GdkEventExpose* event) {
	drawBackgroundSurface();
	m_initialized = true;
	if(data_available) {
		drawTrajectory(m_slipangles, m_currentFrameID); // Draw entire trajectory
	}
	return true;
}


void OrientationTrajectory::drawBackgroundSurface() {
	Glib::RefPtr<Gdk::Window> window = get_window();

	if(window) {
		Gtk::Allocation allocation = get_allocation();
		const int width = allocation.get_width();
		const int height = allocation.get_height();

		// Create new cairo m_surface and context
		m_surface.clear();  // destroy previous m_surface (of possibly different size)
		m_surface = Cairo::ImageSurface::create(Cairo::FORMAT_ARGB32, width, height);
		Cairo::RefPtr<Cairo::Context> cr_surface = Cairo::Context::create(m_surface);

		// Draw empty graph on m_surface
		drawAxes(cr_surface, width, height);

		// Create context for the widget
		Cairo::RefPtr<Cairo::Context> cr = window->create_cairo_context();

		// Show created empty graph
		cr->set_source(m_surface, 0.0, 0.0);
		cr->paint();

	}
}


// Note: y axis is inverted
void OrientationTrajectory::drawAxes(const Cairo::RefPtr<Cairo::Context>& cr, int width, int height) {
	m_axis_range_x = m_axis_limit_x[1]-m_axis_limit_x[0];
	m_axis_range_y = m_axis_limit_y[1]-m_axis_limit_y[0];

	m_padding_left = 0.2*m_axis_range_x;
	m_padding_right = 0.05*m_axis_range_x;
	m_padding_top = 0.05*m_axis_range_y;
	m_padding_bottom = 0.2*m_axis_range_y;

	m_scale_x = width / (m_axis_range_x+m_padding_left+m_padding_right);
	m_scale_y = height / (m_axis_range_y+m_padding_top+m_padding_bottom);
	m_inverse_m_scale_x = 1.0 / m_scale_x;
	m_inverse_m_scale_y = 1.0 / m_scale_y;
	m_offset_x = m_padding_left;
	m_offset_y = m_padding_top + m_axis_limit_y[1];

	//m_line_width = 0.25 * m_scale_x; // normalized scaling
	//m_font_size = 5.0 * m_scale_y; // workspace scaling

	m_line_width = width/300.0;
	m_font_size = width/28.0;


	// Pretty ticks
	numScale.computeScale(m_axis_limit_x[0], m_axis_limit_x[1], 8);
	int num_xticks = numScale.getNumTicks();
	double min_xtick = numScale.getNiceMin();
	double xtick_spacing = numScale.getTickSpacing();

	numScale.computeScale(m_axis_limit_y[0], m_axis_limit_y[1], 8);
	int num_yticks = numScale.getNumTicks();
	double min_ytick = numScale.getNiceMin();
	double ytick_spacing = numScale.getTickSpacing();

	cr->scale(1.0, 1.0); // Scale for uniform lines
	cr->set_line_width(m_line_width);

	// Background
	cr->set_source_rgba(0.85, 0.85, 0.85, 1.0);
	cr->paint();


	//----------
	// Axis box
	//----------

	// Path
	cr->save(); cr->scale(m_scale_x, m_scale_y); cr->translate(m_offset_x, m_offset_y);
	cr->rectangle(m_axis_limit_x[0], -m_axis_limit_y[1], m_axis_range_x, m_axis_range_y); // top left, width, height
	cr->restore();
	// Rendering
	cr->set_line_cap(Cairo::LINE_CAP_SQUARE);
	cr->set_source_rgba(1.0, 1.0, 1.0, 1.0);
	cr->fill_preserve(); // background
	cr->set_source_rgba(0.0, 0.0, 0.0, 1.0);
	cr->stroke(); // outline

	//------------
	// Grid lines
	//------------

	// Path
	cr->save(); cr->scale(m_scale_x, m_scale_y); cr->translate(m_offset_x, m_offset_y);
	// vertical lines
	double tick = min_xtick;
	for(int i = 0; i < num_xticks; i++) {
		cr->move_to(tick, -m_axis_limit_y[1]);
		cr->line_to(tick, -m_axis_limit_y[0]);
		tick += xtick_spacing;
	}
	// horizontal lines
	tick = min_ytick;
	for(int i = 0; i < num_yticks; i++) {
		cr->move_to(m_axis_limit_x[0], -tick);
		cr->line_to(m_axis_limit_x[1], -tick);
		tick += ytick_spacing;
	}
	cr->restore();

	// Rendering
	std::vector<double> dashes(2);
	dashes[0] = 2.0*m_line_width;
	dashes[1] = 2.0*m_line_width;
	cr->set_line_width(m_line_width);
	cr->set_line_cap(Cairo::LINE_CAP_BUTT);
	cr->set_dash(dashes, 0.0);
	cr->set_source_rgba(0.3, 0.3, 0.3, 0.3);
	cr->stroke();

	//------------
	// Axis ticks
	//------------
	cr->set_dash(std::vector<double>(), 0.0); // Disable dashes
	// Path
	cr->save(); cr->scale(m_scale_x, m_scale_y); cr->translate(m_offset_x, m_offset_y);
	// x-ticks
	double tick_length = 8.0*m_inverse_m_scale_y;
	tick = min_xtick+xtick_spacing;
	for(int i = 0; i < num_xticks-1; i++) {
		// bottom
		cr->move_to(tick, -m_axis_limit_y[1]);
		cr->line_to(tick, -m_axis_limit_y[1] + tick_length);
		// top
		cr->move_to(tick, -m_axis_limit_y[0]);
		cr->line_to(tick, -m_axis_limit_y[0] - tick_length);
		tick += xtick_spacing;
	}
	// y-ticks
	tick_length = 8.0 * m_inverse_m_scale_x;
	tick = min_ytick+ytick_spacing;
	for(int i = 0; i < num_yticks-1; i++) {
		// left
		cr->move_to(m_axis_limit_x[0], -tick);
		cr->line_to(m_axis_limit_x[0] + tick_length, -tick);
		// right
		cr->move_to(m_axis_limit_x[1], -tick);
		cr->line_to(m_axis_limit_x[1] - tick_length, -tick);
		tick += ytick_spacing;
	}
	cr->restore();

	// Rendering
	cr->set_line_width(m_line_width);
	cr->set_line_cap(Cairo::LINE_CAP_BUTT);
	cr->set_source_rgba(0.0, 0.0, 0.0, 1.0);
	cr->stroke();


	//-------------
	// Axis labels
	//-------------
	Cairo::RefPtr<Cairo::ToyFontFace> font = Cairo::ToyFontFace::create("sans", Cairo::FONT_SLANT_NORMAL, Cairo::FONT_WEIGHT_NORMAL);	
	std::ostringstream ss;
	std::string valueStr;
	Cairo::TextExtents te_label;
	Cairo::TextExtents te_ticks;

	// x-axis
	valueStr = "t";
	cr->set_font_face(font);
	cr->set_font_size(1.5*m_font_size);
	cr->get_text_extents(valueStr, te_label);
	double offsetX = (0.5 * te_label.width) / m_scale_x;
	double offsetY = (0.5 * te_label.height) / m_scale_y;

	// Path
	cr->save(); cr->scale(m_scale_x, m_scale_y); cr->translate(m_offset_x, m_offset_y);
	//cr->move_to(0.5*m_axis_range_x-offsetX, m_axis_limit_y[1] - 0.8*m_padding_bottom + offsetY);
	cr->move_to(0.5*m_axis_range_x-offsetX, -(m_axis_limit_y[0]-0.8*m_padding_bottom-offsetY) );
	cr->restore();

	// Rendering
	cr->set_source_rgba(0.0, 0.0, 0.0, 1.0);
	cr->show_text(valueStr);

	cr->set_font_face(font);
	cr->set_font_size(m_font_size);
	tick = min_xtick;
	for(int i = 0; i < num_xticks; i++) {
		ss.str("");
		ss << tick;
		valueStr = ss.str();

		cr->get_text_extents(valueStr, te_ticks);
		offsetX = (0.5 * te_ticks.width) / m_scale_x;
		offsetY = (0.5 * te_ticks.height) / m_scale_y;

		// Path
		cr->save(); cr->scale(m_scale_x, m_scale_y); cr->translate(m_offset_x, m_offset_y);
		cr->move_to(tick-offsetX, -(m_axis_limit_y[0]-0.4*m_padding_bottom-offsetY) );
		cr->restore();

		cr->show_text(valueStr);
		tick += xtick_spacing;
	}

	// y-axis
	valueStr = "Θ";
	cr->set_font_face(font);
	cr->set_font_size(1.5*m_font_size);
	cr->get_text_extents(valueStr, te_label);
	offsetX = (0.5 * te_label.width) / m_scale_x;
	offsetY = (0.5 * te_label.height) / m_scale_y;

	// Path
	cr->save(); cr->scale(m_scale_x, m_scale_y); cr->translate(m_offset_x, m_offset_y);
	cr->move_to(m_axis_limit_x[0] - 0.8*m_padding_left - offsetX, -(m_axis_limit_y[0] + 0.5*m_axis_range_y-offsetY) );
	cr->restore();

	// Rendering
	cr->show_text(valueStr);

	cr->set_font_face(font);
	cr->set_font_size(m_font_size);
	tick = min_ytick;
	for(int i = 0; i < num_yticks+1; i++) {
		ss.str(""); // clear
		ss << tick << "°";
		valueStr = ss.str();

		cr->get_text_extents(valueStr, te_ticks);
		offsetX = (0.5 * te_ticks.width) / m_scale_x;
		offsetY = (0.5 * te_ticks.height) / m_scale_y;

		// Path
		cr->save(); cr->scale(m_scale_x, m_scale_y); cr->translate(m_offset_x, m_offset_y);
		cr->move_to(m_axis_limit_x[0] - 0.4*m_padding_left - offsetX, -tick + offsetY);
		cr->restore();

		cr->show_text(valueStr);
		tick += ytick_spacing;
	}

}


bool OrientationTrajectory::updateTrajectory(std::deque<double>& slipangles, uint currentFrameID) {

	Glib::RefPtr<Gdk::Window> window = get_window();
	if(window && m_initialized) {

		m_slipangles = slipangles;
		m_currentFrameID = currentFrameID;
		data_available = true;

		// Check if axis limits are sufficient and adapt them if necessary
		bool redraw = false;
		double angle = slipangles.back();
		m_minAngle = (m_minAngle < angle)? m_minAngle : angle;
		m_maxAngle = (m_maxAngle > angle)? m_maxAngle : angle;
		if(slipangles.size() > m_axis_limit_x[1]-1) {
			m_axis_limit_x[1] *= 2;
			redraw = true;
		}
		if(m_minAngle < m_axis_limit_y[0]) {
			m_axis_limit_y[0] *= 2;
			redraw = true;
		}
		if(m_maxAngle > m_axis_limit_y[1]) {
			m_axis_limit_y[1] *= 2;
			redraw = true;
		}
		if(redraw) {
			drawBackgroundSurface();
			drawTrajectory(slipangles, currentFrameID);
		} else {
			// Draw only last segment of trajectory
			Cairo::RefPtr<Cairo::Context> cr = window->create_cairo_context();
			//cr->set_source_rgba(0.0, 0.1765, 0.4392, 1.0); // uibk blue
			cr->set_source_rgba(1.0, 0.5, 0.0, 1.0); // uibk orange
			cr->set_line_width(2.0*m_line_width);
			cr->set_line_cap(Cairo::LINE_CAP_ROUND);
			cr->save(); cr->scale(m_scale_x, m_scale_y); cr->translate(m_offset_x, m_offset_y);
			uint n = currentFrameID;
			if(n == 0) {
				cr->move_to(0.0, 0.0);
			} else {
				cr->move_to(n-1, -slipangles[n-1]);
			}
			cr->line_to(n, -slipangles[n]);
			cr->restore();
			cr->stroke();
		}
	}
	return true;
}


bool OrientationTrajectory::drawTrajectory(std::deque<double>& slipangles, uint currentFrameID) {

	Glib::RefPtr<Gdk::Window> window = get_window();
	if(window && m_initialized) {

		m_slipangles = slipangles;
		m_currentFrameID = currentFrameID;
		data_available = true;

		// Check if axis limits are sufficient and adapt them if necessary
		bool redraw = false;
		m_minAngle = *min_element(slipangles.begin(), slipangles.end());
		m_maxAngle = *max_element(slipangles.begin(), slipangles.end());

		if(slipangles.size() > m_axis_limit_x[1]-1) {
			m_axis_limit_x[1] *= 2;
			redraw = true;
		}
		if(m_minAngle < m_axis_limit_y[0]) {
			m_axis_limit_y[0] *= 2;
			redraw = true;
		}
		if(m_maxAngle > m_axis_limit_y[1]) {
			m_axis_limit_y[1] *= 2;
			redraw = true;
		}
		if(redraw) {
			drawBackgroundSurface();
		}

		// Create cairo context for the widget
		Cairo::RefPtr<Cairo::Context> cr = window->create_cairo_context();

		// Draw empty graph
		cr->set_source(m_surface, 0.0, 0.0);
		cr->paint();

		//---------------------------
		// Rotation angle trajectory
		//---------------------------
		//cr->set_source_rgba(0.0, 0.1765, 0.4392, 1.0); // uibk blue
		cr->set_source_rgba(1.0, 0.5, 0.0, 1.0); // uibk orange
		cr->set_line_width(2.0*m_line_width);
		cr->set_line_cap(Cairo::LINE_CAP_ROUND);
		// Path
		cr->save(); cr->scale(m_scale_x, m_scale_y); cr->translate(m_offset_x, m_offset_y);
		cr->move_to(0, slipangles[0]);
		for(uint i = 1; i <= currentFrameID; i++) {
			cr->line_to(i, -slipangles[i]);
		}
		cr->restore();
		cr->stroke();

	}
	return true;
}


// ----------------------------------------------------------------------------
// Multiplot
// ----------------------------------------------------------------------------

guiSlipDetectionMultiPlot::guiSlipDetectionMultiPlot()
: // Aspect ratios: 1.0
			  m_AspectFrame_UpperLeft("Current slip-vector", Gtk::ALIGN_CENTER, Gtk::ALIGN_CENTER, 1.0, false),
			  m_AspectFrame_UpperRight("Current orientation", Gtk::ALIGN_CENTER, Gtk::ALIGN_CENTER, 1.0, false),
			  m_AspectFrame_LowerLeft("Translation over time", Gtk::ALIGN_CENTER, Gtk::ALIGN_CENTER, 1.0, false),
			  m_AspectFrame_LowerRight("Rotation over time", Gtk::ALIGN_CENTER, Gtk::ALIGN_CENTER, 1.0, false)
{

	set_title("Slippery when wet!");
	set_border_width(0);
	set_size_request(450, 450);

	set_skip_taskbar_hint(true); // No task bar entry
	set_type_hint(Gdk::WINDOW_TYPE_HINT_DIALOG); // Always on top

	Gdk::Geometry geometry = { -1, -1,     // Min dimensions
							   -1, -1,     // Max dimensions
							   -1, -1,     // Base dimension
							   -1, -1,     // Increment
							   1.0, 1.0 }; // Min/Max aspect ratio (r = width/height)
	set_geometry_hints(*this, geometry, Gdk::HINT_ASPECT); // Only pay attention to aspect ratio

	m_slipVectorLive = boost::shared_ptr<SlipVectorLive>(new SlipVectorLive());
	m_HBox_UpperLeft.pack_start(*m_slipVectorLive, Gtk::PACK_EXPAND_WIDGET, 0);
	m_HBox_UpperLeft.set_border_width(5);
	m_AspectFrame_UpperLeft.add(m_HBox_UpperLeft);
	m_AspectFrame_UpperLeft.set_border_width(5);

	m_orientation = boost::shared_ptr<Orientation>(new Orientation());
	m_HBox_UpperRight.pack_start(*m_orientation, Gtk::PACK_EXPAND_WIDGET, 0);
	m_HBox_UpperRight.set_border_width(5);
	m_AspectFrame_UpperRight.add(m_HBox_UpperRight);
	m_AspectFrame_UpperRight.set_border_width(5);

	m_slipvectors_dummy.push_back( boost::make_tuple(0.0, 0.0) );
	m_slipVectorTrajectory = boost::shared_ptr<SlipVectorTrajectory>(new SlipVectorTrajectory(m_slipvectors_dummy));
	m_HBox_LowerLeft.pack_start(*m_slipVectorTrajectory, Gtk::PACK_EXPAND_WIDGET, 0);
	m_HBox_LowerLeft.set_border_width(5);
	m_AspectFrame_LowerLeft.add(m_HBox_LowerLeft);
	m_AspectFrame_LowerLeft.set_border_width(5);

	m_slipangles_dummy.push_back(0.0);
	m_orientationTrajectory = boost::shared_ptr<OrientationTrajectory>(new OrientationTrajectory(m_slipangles_dummy));
	m_HBox_LowerRight.pack_start(*m_orientationTrajectory, Gtk::PACK_EXPAND_WIDGET, 0);
	m_HBox_LowerRight.set_border_width(5);
	m_AspectFrame_LowerRight.add(m_HBox_LowerRight);
	m_AspectFrame_LowerRight.set_border_width(5);

	m_HBox_Upper.pack_start(m_AspectFrame_UpperLeft, Gtk::PACK_EXPAND_WIDGET, 0);
	m_HBox_Upper.pack_start(m_AspectFrame_UpperRight, Gtk::PACK_EXPAND_WIDGET, 0);
	m_HBox_Lower.pack_start(m_AspectFrame_LowerLeft, Gtk::PACK_EXPAND_WIDGET, 0);
	m_HBox_Lower.pack_start(m_AspectFrame_LowerRight, Gtk::PACK_EXPAND_WIDGET, 0);
	m_VBox_Main.pack_start(m_HBox_Upper);
	m_VBox_Main.pack_start(m_HBox_Lower);

	add(m_VBox_Main);
	show_all();
	}


guiSlipDetectionMultiPlot::~guiSlipDetectionMultiPlot() { }


void guiSlipDetectionMultiPlot::reset() {
	m_slipVectorLive->reset();
	m_orientation->reset();
	m_slipVectorTrajectory->reset();
	m_orientationTrajectory->reset();
}


void guiSlipDetectionMultiPlot::setAxisLimits(int x_lower, int x_upper, int y_lower,  int y_upper) {
	m_orientationTrajectory->resetAxisLimits(x_lower, x_upper, y_lower, y_upper);
	m_orientationTrajectory->drawBackgroundSurface();
}


// Draw full path up to current frame
void guiSlipDetectionMultiPlot::drawTrajectory(slipResult& slip,
		                                       std::deque<slip_trajectory>& slipvectors,
		                                       std::deque<double>& slipangles,
		                                       uint currentFrameID) {

	m_slipVectorLive->drawVector(slip.slipVector_x, slip.slipVector_y);
	m_orientation->drawOrientation(slip.successRotation, slip.orientation, slip.lambda1, slip.lambda2, slip.skew_x, slip.skew_y);
	m_slipVectorTrajectory->drawTrajectory(slipvectors, currentFrameID);
	m_orientationTrajectory->drawTrajectory(slipangles, currentFrameID);
}


// Draw full path up to current frame
void guiSlipDetectionMultiPlot::drawTrajectoryReference(slipResult& slip,
		                                       std::deque<slip_trajectory>& slipvectors,
		                                       std::deque<double>& slipangles,
		                                       uint currentFrameID) {

	m_slipVectorTrajectory->drawTrajectory(slipvectors, currentFrameID);
	m_orientationTrajectory->drawTrajectory(slipangles, currentFrameID);
}


// Only draw last segment of path (assumes drawing of consecutive frames)
void guiSlipDetectionMultiPlot::updateTrajectory(slipResult &slip,
                                                 std::deque<slip_trajectory>& slipvectors,
                                                 std::deque<double>& slipangles,
                                                 uint currentFrameID) {

	m_slipVectorLive->drawVector(slip.slipVector_x, slip.slipVector_y);
	m_orientation->drawOrientation(slip.successRotation, slip.orientation, slip.lambda1, slip.lambda2, slip.skew_x, slip.skew_y);
	m_slipVectorTrajectory->updateTrajectory(slipvectors, currentFrameID);
	m_orientationTrajectory->updateTrajectory(slipangles, currentFrameID);
}
