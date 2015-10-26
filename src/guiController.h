#ifndef GUICONTROLLER_H_
#define GUICONTROLLER_H_

#include <gtkmm.h>

#include "controller.h"

// Forward declarations
class Controller;
class FrameManager;
class FrameProcessor;
class FrameGrabberDSA;
class FrameGrabberSDH;

/**
 * @class guiController
 * @brief GUI to control the SDH-2
 */
class guiController : public Gtk::Frame {

public:
	guiController(Controller *c);
	virtual ~guiController();

	void connectSDH();
	void disconnectSDH();

	void connectDSA();
	void disconnectDSA();

protected:

	Controller *controller;
	FrameManager *frameManager;
	FrameProcessor *frameProcessor;
	FrameGrabberDSA *frameGrabberDSA;
	FrameGrabberSDH *frameGrabberSDH;

	// Images
	Gtk::Image m_Image_Play_SDH;
	Gtk::Image m_Image_Pause_SDH;
	Gtk::Image m_Image_Stop_SDH;
	Gtk::Image m_Image_Record_SDH;
	Gtk::Image m_Image_Play_DSA;
	Gtk::Image m_Image_Pause_DSA;
	Gtk::Image m_Image_Stop_DSA;
	Gtk::Image m_Image_Record_DSA;

	// Basic layout
	Gtk::VBox m_VBox_Left_Sidebar;
	Gtk::Frame m_Frame_SDH;
	Gtk::Frame m_Frame_DSA;
	Gtk::VBox m_VBox_SDH;
	Gtk::VBox m_VBox_DSA;

	// Recorders
	Gtk::Frame m_Frame_Recorder_SDH;
	Gtk::Frame m_Frame_Recorder_DSA;
	Gtk::VBox m_VBox_Recorder_SDH;
	Gtk::VBox m_VBox_Recorder_DSA;

	// Buttons SDH Recorder
	Gtk::HButtonBox m_ButtonBox_Recording_SDH;
	Gtk::Button m_Button_Pause_SDH;
	Gtk::Button m_Button_Stop_SDH;
	Gtk::Button m_Button_Record_SDH;
	bool recorder_paused_SDH;
	bool recorder_recording_SDH;
	Gtk::CheckButton m_CheckButton_Temperature;
	Gtk::CheckButton m_CheckButton_JointAngles;

	// Buttons DSA Recorder
	Gtk::HButtonBox m_ButtonBox_Recording_DSA;
	Gtk::Button m_Button_Pause_DSA;
	Gtk::Button m_Button_Stop_DSA;
	Gtk::Button m_Button_Record_DSA;
	bool recorder_paused_DSA;
	bool recorder_recording_DSA;

	// Preshape Grasp selection
	Gtk::Frame m_Frame_Grasp;
	Gtk::VBox m_VBox_Grasp;
	Gtk::HBox m_HBox_Grasp;
	Gtk::ComboBoxText m_Combo_Grasp;
	Gtk::Button m_Button_Grasp;
	Gtk::Button m_Button_Grasp_Reactive;
	bool m_ToggleButton_Grasp_Slip_Failed;
	Gtk::ToggleButton m_ToggleButton_Grasp_Slip;
	Gtk::Button m_Button_Relax;

	// Normal grasp
	Glib::Thread *m_Thread_Grasp;
	Glib::Dispatcher m_Thread_Grasp_Dispatcher;
	void worker_grasp(int graspID, double closeRatio, double velocity);
	void on_worker_grasp_done();

	// Reactive grasp
	Glib::Thread *m_Thread_Grasp_Reactive;
	Glib::Dispatcher m_Thread_Grasp_Reactive_Dispatcher;
	void worker_grasp_reactive(int graspID, double velocity, double limit);
	void on_worker_grasp_reactive_done();

	// Reactive grasp with slip detection
	bool m_stop_thread_grasp_slip;
	Glib::Mutex m_mutex_thread_grasp_slip;
	Glib::Thread *m_Thread_Grasp_Slip;
	Glib::Dispatcher m_Thread_Grasp_Slip_Dispatcher;
	void worker_grasp_slip(int graspID, double velocity, double limitLow, double limitHigh, double stepSize);
	void on_worker_grasp_slip_done();

	// Close-Ratio Slider
	Gtk::Frame m_Frame_Close;
	Gtk::Adjustment m_Adjustment_Close;
	Gtk::HScale m_Slider_Close;

	// Velocity Slider
	Gtk::Frame m_Frame_Velocity;
	Gtk::Adjustment m_Adjustment_Velocity;
	Gtk::HScale m_Slider_Velocity;

	// Reactive grasping slider
	Gtk::Frame m_Frame_Reactive;
	Gtk::Adjustment m_Adjustment_Reactive;
	Gtk::HScale m_Slider_Reactive;

	// Sensitivity Slider
	Gtk::Frame m_Frame_Sensitivity;
	Gtk::Adjustment m_Adjustment_Sensitivity;
	Gtk::HScale m_Slider_Sensitivity;
	Gtk::Label m_Label_Sensitivity;
	float m_sensitivity;

	// Threshold Slider
	Gtk::Frame m_Frame_Threshold;
	Gtk::VBox m_VBox_Threshold;
	Gtk::Adjustment m_Adjustment_Threshold;
	Gtk::HScale m_Slider_Threshold;
	Gtk::Label m_Label_Threshold;
	UInt16 m_threshold;
	Gtk::HButtonBox m_ButtonBox_Threshold;
	Gtk::Button m_Button_Threshold;
	Gtk::Button m_Button_Threshold_Reset;

	// Signal handlers:
	void on_button_pause_SDH_clicked();
	void on_button_stop_SDH_clicked();
	void on_button_record_SDH_clicked();

	void on_checkbutton_temperature_clicked();
	void on_checkbutton_joint_angles_clicked();

	void on_button_pause_DSA_clicked();
	void on_button_stop_DSA_clicked();
	void on_button_record_DSA_clicked();

	void on_combo_grasp_changed();
	void on_button_grasp_clicked();
	void on_button_grasp_reactive_clicked();
	void on_button_grasp_slip_clicked();
	void on_button_relax_clicked();

	bool on_slider_close_clicked(GdkEventButton* event);
	bool on_slider_close_released(GdkEventButton* event);
	bool on_slider_close_value_changed(Gtk::ScrollType type, double value);

	bool on_slider_velocity_clicked(GdkEventButton* event);
	bool on_slider_velocity_released(GdkEventButton* event);
	bool on_slider_velocity_value_changed(Gtk::ScrollType type, double value);

	bool on_slider_reactive_clicked(GdkEventButton* event);
	bool on_slider_reactive_released(GdkEventButton* event);
	bool on_slider_reactive_value_changed(Gtk::ScrollType type, double value);

	bool on_slider_sensitivity_clicked(GdkEventButton* event);
	bool on_slider_sensitivity_released(GdkEventButton* event);
	bool on_slider_sensitivity_value_changed(Gtk::ScrollType type, double value);

	bool on_slider_threshold_clicked(GdkEventButton* event);
	bool on_slider_threshold_released(GdkEventButton* event);
	bool on_slider_threshold_value_changed(Gtk::ScrollType type, double value);
	void on_button_threshold_clicked();
	void on_button_threshold_reset_clicked();
};

#endif /* GUISDH_H_ */
