#ifndef GUISLIPDETECTION_H_
#define GUISLIPDETECTION_H_

#include <deque>

#include <gtkmm.h>

#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>

#include "controller.h"
#include "guiSlipDetectionMultiPlot.h"

class Controller;
class guiMain;

/**
 * @class guiSlipDetection
 * @brief The Slip-detection control GUI.
 */
class guiSlipDetection : public Gtk::Window {

private:

	Controller *controller;
	FrameManager *frameManager;
	FrameProcessor *frameProcessor;
	guiMain *mainGUI;

	sigc::connection m_connection; // Signal

	Gtk::Image m_Image_grey;
	Gtk::Image m_Image_green;
	Gtk::Image m_Image_amber;
	Gtk::Image m_Image_red;

	Gtk::Frame m_Frame_Threshold_Reference;
	Gtk::Adjustment m_Adjustment_Threshold_Reference;
	Gtk::HScale m_Slider_Threshold_Reference;

	Gtk::Frame m_Frame_Threshold_Consecutive;
	Gtk::Adjustment m_Adjustment_Threshold_Consecutive;
	Gtk::HScale m_Slider_Threshold_Consecutive;

	std::vector<Gtk::CheckButton*> m_checkbuttons;
	std::vector<Gtk::Button*> m_buttons;
	std::vector<Gtk::ToggleButton*> m_togglebuttons;
	std::vector<Gtk::HBox*> m_statushboxes;
	std::vector<Gtk::Label*> m_statuslabels;
	std::vector<Gtk::Image*> m_statusimages;
	std::vector<Gtk::VBox*> m_vboxes;
	std::vector<Gtk::Frame*> m_frames;
	Gtk::Table m_table; // Note: Gtk::Table is deprecated in GTK 3, use Gtk::Grid instead

	Gtk::VBox m_VBox_Main;

	std::vector<guiSlipDetectionMultiPlot*> m_multiplots;

	slipResult m_slip;
	std::vector<boost::shared_ptr<std::deque<slipResult> > > m_slipResults;
	std::vector<boost::shared_ptr<std::deque<slip_trajectory> > > m_slipvectors;
	std::vector<boost::shared_ptr<std::deque<double> > > m_slipangles;

	std::vector<uint> m_startFrame;
	std::vector<uint> m_stopFrame;

	double m_threshSlipReference;
	double m_threshSlipConsecutive;
	double m_threshSlipAngle;

	enum connectionMode { OFFLINE, ONLINE, INACTIVE };
	connectionMode m_mode;

	enum blinkenLight { GREY, GREEN, AMBER, RED };
	std::vector<blinkenLight> m_blinkenlight_actual;
	std::vector<blinkenLight> m_blinkenlight_target;

	void setBlinkenLights();

public:
	guiSlipDetection(Controller *c, guiMain *gui);
	virtual ~guiSlipDetection();

	void clearTrajectory(uint m);

	void setModeOnline();
	void setModeOffline();

	bool runSlipDetectionOnline();
	bool drawTrajectoryOnline(uint m);

	void setCurrentFrameOffline(uint frameID);
	void runSlipDetectionOffline(uint m, uint startFrame, uint stopFrame);
	bool drawTrajectoryOffline(uint m, uint currentFrameID);

protected:

	bool on_slider_threshold_reference_value_changed(Gtk::ScrollType type, double value);
	bool on_slider_threshold_reference_clicked(GdkEventButton* event);
	bool on_slider_threshold_reference_released(GdkEventButton* event);

	bool on_slider_threshold_consecutive_value_changed(Gtk::ScrollType type, double value);
	bool on_slider_threshold_consecutive_clicked(GdkEventButton* event);
	bool on_slider_threshold_consecutive_released(GdkEventButton* event);

	void on_checkbutton_enable_clicked(uint m);
	void on_button_set_reference_clicked(uint m);
	void on_togglebutton_details_clicked(uint m);
	bool on_delete_detail_clicked(GdkEventAny* event, uint m);
};

#endif /* GUISLIPDETECTION_H_ */
