#ifndef GUISLIPDETECTIONMULTIPLOT_H_
#define GUISLIPDETECTIONMULTIPLOT_H_

#include <gtkmm.h>
#include <gtkmm/drawingarea.h>
#include <cairomm/context.h>

#include <boost/tuple/tuple.hpp> 

#include "framemanager.h"
#include "slipdetection.h"
#include "utils.h"

typedef boost::tuple<double, double> slip_trajectory;


/**
 * @class NiceScale
 * @brief Pretty axis tick labels. Graphics Gems, Volume 1 by Andrew S. Glassner
 */
class NiceScale {

private:
	double m_tickSpacing;
	double m_range;
	double m_niceMin;
	double m_niceMax;
	int m_numTicks;

	double niceNum(double range, bool round);

public:

	NiceScale();
	NiceScale(double min, double max, int maxTicks);
	virtual ~NiceScale();

	void computeScale(double min, double max, int maxTicks);
	double getNiceMin();
	double getNiceMax();
	double getTickSpacing();
	int getNumTicks();
};


/**
 * @class SlipVectorLive
 * @brief Visualizes current slip vector
 */
class SlipVectorLive : public Gtk::DrawingArea {
private:
	bool m_initialized;
	double m_axis_limit_x[2];
	double m_axis_limit_y[2];
	double m_axis_range_x;
	double m_axis_range_y;
	double m_padding_left;
	double m_padding_right;
	double m_padding_bottom;
	double m_padding_top;
	double m_line_width;
	double m_scale_x;
	double m_scale_y;
	double m_offset_x;
	double m_offset_y;
	Cairo::RefPtr<Cairo::ImageSurface> m_surface; // Empty graph

	bool data_available;
	double m_x, m_y;

public:
	SlipVectorLive();
	virtual ~SlipVectorLive();

	void reset();
	void drawAxes(const Cairo::RefPtr<Cairo::Context>& cr, int width, int height);
	bool drawVector(double x, double y);

protected:
	//Override default signal handler:
	virtual bool on_expose_event(GdkEventExpose* event);
};


/**
 * @class SlipVectorTrajectory
 * @brief Visualizes slip vector trajectory
 */
class SlipVectorTrajectory : public Gtk::DrawingArea {
private:
	bool m_initialized;
	double m_axis_limit_x[2];
	double m_axis_limit_y[2];
	double m_axis_range_x;
	double m_axis_range_y;
	double m_padding_left;
	double m_padding_right;
	double m_padding_bottom;
	double m_padding_top;
	double m_horizontal_spacer;
	double m_line_width;
	double m_scale_x;
	double m_scale_y;
	double m_offset_x;
	double m_offset_y;
	Cairo::RefPtr<Cairo::ImageSurface> m_surface; // Empty graph

	bool data_available;
	std::deque<slip_trajectory>& m_slipvectors;
	uint m_currentFrameID;

public:
	SlipVectorTrajectory(std::deque<slip_trajectory>& slipvectors);
	virtual ~SlipVectorTrajectory();

	void reset();
	void drawBackgroundSurface();
	void drawAxes(const Cairo::RefPtr<Cairo::Context>& cr, int width, int height);
	bool drawTrajectory(std::deque<slip_trajectory>& slipvectors, uint currentFrameID);
	bool updateTrajectory(std::deque<slip_trajectory>& slipvectors, uint currentFrameID);

protected:
	//Override default signal handler:
	virtual bool on_expose_event(GdkEventExpose* event);
};


/**
 * @class Orientation
 * @brief Visualizes current orientation
 */
class Orientation : public Gtk::DrawingArea {
private:
	bool m_initialized;
	double m_axis_limit_x[2];
	double m_axis_limit_y[2];
	double m_axis_range_x;
	double m_axis_range_y;
	double m_padding_left;
	double m_padding_right;
	double m_padding_bottom;
	double m_padding_top;
	double m_line_width;
	double m_scale_x;
	double m_scale_y;
	double m_offset_x;
	double m_offset_y;
	double m_skew_x_pos;
	double m_skew_y_pos;
	double m_max_skew_x;
	double m_max_skew_y;
	Cairo::RefPtr<Cairo::ImageSurface> m_surface; // Empty graph

	bool data_available;
	bool m_success;
	double m_angle, m_lambda1, m_lambda2, m_skew_x, m_skew_y;

public:
	Orientation();
	virtual ~Orientation();

	void reset();
	void drawAxes(const Cairo::RefPtr<Cairo::Context>& cr, int width, int height);
	bool drawOrientation(bool success, double angle, double lambda1, double lambda2, double skew_x, double skew_y);

protected:
	//Override default signal handler:
	virtual bool on_expose_event(GdkEventExpose* event);
};


/**
 * @class OrientationTrajectory
 * @brief Visualizes rotation trajectory
 */
class OrientationTrajectory : public Gtk::DrawingArea {
private:
	bool m_initialized;
	double m_axis_limit_x[2];
	double m_axis_limit_y[2];
	double m_axis_range_x;
	double m_axis_range_y;
	double m_padding_left;
	double m_padding_right;
	double m_padding_bottom;
	double m_padding_top;
	double m_vertical_spacer;
	double m_line_width;
	double m_font_size;
	double m_scale_x;
	double m_scale_y;
	double m_offset_x;
	double m_offset_y;
	double m_inverse_m_scale_x;
	double m_inverse_m_scale_y;
	double m_minAngle;
	double m_maxAngle;

	Cairo::RefPtr<Cairo::ImageSurface> m_surface; // Empty graph
	NiceScale numScale;

	bool data_available;
	std::deque<double>& m_slipangles;
	uint m_currentFrameID;
public:
	OrientationTrajectory(std::deque<double>& slipangles);
	virtual ~OrientationTrajectory();

	void reset();
	void resetAxisLimits(int x_lower, int x_upper, int y_lower,  int y_upper);
	void drawBackgroundSurface();
	void drawAxes(const Cairo::RefPtr<Cairo::Context>& cr, int width, int height);
	bool drawTrajectory(std::deque<double>& slipangles, uint currentFrameID);
	bool updateTrajectory(std::deque<double>& slipangles, uint currentFrameID);

protected:
	//Override default signal handler:
	virtual bool on_expose_event(GdkEventExpose* event);
};


/**
 * @class guiSlipDetectionMultiPlot
 * @brief Combines individual widgets.
 */
class guiSlipDetectionMultiPlot : public virtual Gtk::Window {

private:

	FrameManager *frameManager;

	boost::shared_ptr<SlipVectorLive> m_slipVectorLive;
	boost::shared_ptr<Orientation> m_orientation;
	boost::shared_ptr<SlipVectorTrajectory> m_slipVectorTrajectory;
	boost::shared_ptr<OrientationTrajectory> m_orientationTrajectory;

	std::deque<double> m_slipangles_dummy;
	std::deque<slip_trajectory> m_slipvectors_dummy;

public:
	guiSlipDetectionMultiPlot();
	virtual ~guiSlipDetectionMultiPlot();

	void drawTrajectory(slipResult &slip,
			std::deque<slip_trajectory>& slipvectors,
			std::deque<double>& slipangles,
			uint currentFrameID);

	void updateTrajectory(slipResult &slip,
			std::deque<slip_trajectory>& slipvectors,
			std::deque<double>& slipangles,
			uint currentFrameID);

	void drawTrajectoryReference(slipResult &slip,
			std::deque<slip_trajectory>& slipvectors,
			std::deque<double>& slipangles,
			uint currentFrameID);

	void reset();
	void setAxisLimits(int x_lower, int x_upper, int y_lower,  int y_upper);

protected:

	Gtk::VBox m_VBox_Main;
	Gtk::HBox m_HBox_Upper;
	Gtk::HBox m_HBox_Lower;

	Gtk::HBox m_HBox_UpperLeft;
	Gtk::HBox m_HBox_UpperRight;
	Gtk::HBox m_HBox_LowerLeft;
	Gtk::HBox m_HBox_LowerRight;

	Gtk::AspectFrame m_AspectFrame_UpperLeft;
	Gtk::AspectFrame m_AspectFrame_UpperRight;
	Gtk::AspectFrame m_AspectFrame_LowerLeft;
	Gtk::AspectFrame m_AspectFrame_LowerRight;

};

#endif /* GUISLIPDETECTIONMULTIPLOT_H_ */
