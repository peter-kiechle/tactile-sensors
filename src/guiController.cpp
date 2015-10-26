#include "guiController.h"
#include "utils.h"

guiController::guiController(Controller *c)
:
m_Adjustment_Close(0.5, 0.0, 1.0, 1.0, 0.0, 0.0), // initial, lower, upper, step_increment, page_increment(pointless), page_size(pointless), max value = upper - page_size
m_Slider_Close(m_Adjustment_Close),
m_Adjustment_Velocity(10.0, 1.0, 100.0, 1.0, 0.0, 0.0),
m_Slider_Velocity(m_Adjustment_Velocity),
m_Adjustment_Reactive(200.0, 3.0, 3600.0, 100.0, 0.0, 0.0),
m_Slider_Reactive(m_Adjustment_Reactive),
m_Adjustment_Sensitivity(0.5, 0.0, 1.0, 0.1, 0.0, 0.0),
m_Slider_Sensitivity(m_Adjustment_Sensitivity),
m_Adjustment_Threshold(150.0, 0.0, 200.0, 1.0, 0.0, 0.0),
m_Slider_Threshold(m_Adjustment_Threshold)
{
	controller = c;
	frameManager = controller->getFrameManager();
	frameProcessor = frameManager->getFrameProcessor();

	set_shadow_type(Gtk::SHADOW_NONE);

	// Images
	m_Image_Play_SDH.set(Gtk::Stock::MEDIA_PLAY, Gtk::ICON_SIZE_BUTTON); // ICON_SIZE_MENU / ICON_SIZE_BUTTON
	m_Image_Pause_SDH.set(Gtk::Stock::MEDIA_PAUSE, Gtk::ICON_SIZE_BUTTON);
	m_Image_Stop_SDH.set(Gtk::Stock::MEDIA_STOP, Gtk::ICON_SIZE_BUTTON);
	m_Image_Record_SDH.set(Gtk::Stock::MEDIA_RECORD, Gtk::ICON_SIZE_BUTTON);

	m_Image_Play_DSA.set(Gtk::Stock::MEDIA_PLAY, Gtk::ICON_SIZE_BUTTON); // ICON_SIZE_MENU / ICON_SIZE_BUTTON
	m_Image_Pause_DSA.set(Gtk::Stock::MEDIA_PAUSE, Gtk::ICON_SIZE_BUTTON);
	m_Image_Stop_DSA.set(Gtk::Stock::MEDIA_STOP, Gtk::ICON_SIZE_BUTTON);
	m_Image_Record_DSA.set(Gtk::Stock::MEDIA_RECORD, Gtk::ICON_SIZE_BUTTON);

	//--------------------
	// SDH Recorder Frame
	//--------------------
	m_Frame_Recorder_SDH.set_label("Frame grabber SDH");
	m_Frame_Recorder_SDH.set_label_align(0.02, 0.5);
	m_Frame_Recorder_SDH.set_sensitive(false);

	// Buttons SDH Recorder
	m_Button_Pause_SDH.set_image(m_Image_Play_SDH);
	m_Button_Pause_SDH.signal_clicked().connect(sigc::mem_fun(*this, &guiController::on_button_pause_SDH_clicked));
	m_Button_Record_SDH.set_image(m_Image_Record_SDH);
	m_Button_Record_SDH.set_sensitive(true);
	m_Button_Record_SDH.signal_clicked().connect(sigc::mem_fun(*this, &guiController::on_button_record_SDH_clicked));
	m_Button_Stop_SDH.set_image(m_Image_Stop_SDH);
	m_Button_Stop_SDH.set_sensitive(false);
	m_Button_Stop_SDH.signal_clicked().connect(sigc::mem_fun(*this, &guiController::on_button_stop_SDH_clicked));
	m_ButtonBox_Recording_SDH.pack_start(m_Button_Pause_SDH, Gtk::PACK_SHRINK, 5);
	m_ButtonBox_Recording_SDH.pack_start(m_Button_Record_SDH, Gtk::PACK_SHRINK, 5);
	m_ButtonBox_Recording_SDH.pack_start(m_Button_Stop_SDH, Gtk::PACK_SHRINK, 5);
	m_ButtonBox_Recording_SDH.set_border_width(5);

	m_CheckButton_Temperature.set_label("Temperatures");
	m_CheckButton_Temperature.set_active(true);
	m_CheckButton_Temperature.signal_clicked().connect( sigc::mem_fun(*this, &guiController::on_checkbutton_temperature_clicked) );

	m_CheckButton_JointAngles.set_label("Joint angles");
	m_CheckButton_JointAngles.set_active(true);
	m_CheckButton_JointAngles.signal_clicked().connect( sigc::mem_fun(*this, &guiController::on_checkbutton_joint_angles_clicked) );


	//--------------------
	// DSA Recorder Frame
	//--------------------
	m_Frame_Recorder_DSA.set_label("Frame grabber DSA");
	m_Frame_Recorder_DSA.set_label_align(0.02, 0.5);
	m_Frame_Recorder_DSA.set_sensitive(false);

	// Buttons DSA Recorder
	m_Button_Pause_DSA.set_image(m_Image_Play_DSA);
	m_Button_Pause_DSA.set_sensitive(false);
	m_Button_Pause_DSA.signal_clicked().connect(sigc::mem_fun(*this, &guiController::on_button_pause_DSA_clicked));
	m_Button_Record_DSA.set_image(m_Image_Record_DSA);
	m_Button_Record_DSA.set_sensitive(false);
	m_Button_Record_DSA.signal_clicked().connect(sigc::mem_fun(*this, &guiController::on_button_record_DSA_clicked));
	m_Button_Stop_DSA.set_image(m_Image_Stop_DSA);
	m_Button_Stop_DSA.set_sensitive(false);
	m_Button_Stop_DSA.signal_clicked().connect(sigc::mem_fun(*this, &guiController::on_button_stop_DSA_clicked));
	m_ButtonBox_Recording_DSA.pack_start(m_Button_Pause_DSA, Gtk::PACK_SHRINK, 5);
	m_ButtonBox_Recording_DSA.pack_start(m_Button_Record_DSA, Gtk::PACK_SHRINK, 5);
	m_ButtonBox_Recording_DSA.pack_start(m_Button_Stop_DSA, Gtk::PACK_SHRINK, 5);
	m_ButtonBox_Recording_DSA.set_border_width(5);

	// Recorder assembly
	m_VBox_Recorder_SDH.pack_start(m_ButtonBox_Recording_SDH, Gtk::PACK_SHRINK, 0);
	m_VBox_Recorder_SDH.pack_start(m_CheckButton_Temperature, Gtk::PACK_SHRINK, 0);
	m_VBox_Recorder_SDH.pack_start(m_CheckButton_JointAngles, Gtk::PACK_SHRINK, 0);
	m_Frame_Recorder_SDH.add(m_VBox_Recorder_SDH);

	m_VBox_Recorder_DSA.pack_start(m_ButtonBox_Recording_DSA, Gtk::PACK_SHRINK, 0);
	m_Frame_Recorder_DSA.add(m_VBox_Recorder_DSA);


	//--------------------
	// SDH Frame
	//--------------------
	m_Frame_SDH.set_label("Joint Controller");
	m_Frame_SDH.set_label_align(0.02, 0.5);
	m_Frame_SDH.set_sensitive(false);

	// Grasp Frame
	m_Combo_Grasp.append("Parallel - 0°, parallel fingertips");
	m_Combo_Grasp.append("Cylindrical - 0°, tilted fingertips");
	m_Combo_Grasp.append("Centrical - 60°, parallel fingertips");
	m_Combo_Grasp.append("Spherical - 60°, tilted fingertips");
	m_Combo_Grasp.append("Pinch - 90° parallel fingertips");
	m_Combo_Grasp.append("Pinch - 90° tilted fingertips");
	m_Combo_Grasp.append("Experimental");
	m_Combo_Grasp.set_active(4); // Initial value
	m_Combo_Grasp.set_size_request(120);
	m_Combo_Grasp.signal_changed().connect(sigc::mem_fun(*this, &guiController::on_combo_grasp_changed));

	m_Button_Relax.set_label("Relax");
	m_Button_Relax.signal_clicked().connect(sigc::mem_fun(*this, &guiController::on_button_relax_clicked));
	m_Button_Grasp.set_label("Grasp");
	m_Button_Grasp.set_size_request(-1, 36);
	m_Button_Grasp.signal_clicked().connect(sigc::mem_fun(*this, &guiController::on_button_grasp_clicked));
	m_Button_Grasp_Reactive.set_label("Tactile");
	m_Button_Grasp_Reactive.signal_clicked().connect(sigc::mem_fun(*this, &guiController::on_button_grasp_reactive_clicked));
	m_ToggleButton_Grasp_Slip.set_label("Adaptive");
	m_ToggleButton_Grasp_Slip.signal_clicked().connect(sigc::mem_fun(*this, &guiController::on_button_grasp_slip_clicked));
	m_ToggleButton_Grasp_Slip_Failed = false;


	// Grasp threads
	m_Thread_Grasp = NULL;
	m_Thread_Grasp_Reactive = NULL;
	m_Thread_Grasp_Slip = NULL;
	m_Thread_Grasp_Dispatcher.connect(sigc::mem_fun(*this, &guiController::on_worker_grasp_done));
	m_Thread_Grasp_Reactive_Dispatcher.connect(sigc::mem_fun(*this, &guiController::on_worker_grasp_reactive_done));
	m_Thread_Grasp_Slip_Dispatcher.connect(sigc::mem_fun(*this, &guiController::on_worker_grasp_slip_done));
	m_stop_thread_grasp_slip = false;

	m_HBox_Grasp.pack_start(m_Button_Relax, Gtk::PACK_EXPAND_WIDGET, 0);
	m_HBox_Grasp.pack_start(m_Button_Grasp, Gtk::PACK_EXPAND_WIDGET, 0);
	m_HBox_Grasp.pack_start(m_Button_Grasp_Reactive, Gtk::PACK_EXPAND_WIDGET, 0);
	m_HBox_Grasp.pack_start(m_ToggleButton_Grasp_Slip, Gtk::PACK_EXPAND_WIDGET, 0);

	m_VBox_Grasp.pack_start(m_Combo_Grasp, Gtk::PACK_SHRINK, 0);
	m_VBox_Grasp.pack_start(*Gtk::manage(new Gtk::HSeparator()), Gtk::PACK_EXPAND_WIDGET, 5);
	m_VBox_Grasp.pack_start(m_HBox_Grasp, Gtk::PACK_SHRINK, 0);
	m_VBox_Grasp.set_border_width(0);
	m_Frame_Grasp.add(m_VBox_Grasp);

	m_Frame_Grasp.set_label("Grasp Preshape");
	m_Frame_Grasp.set_label_align(0.02, 0.5);
	m_Frame_Grasp.set_border_width(5);
	m_Frame_Grasp.set_shadow_type(Gtk::SHADOW_NONE);

	// Close ratio slider
	m_Slider_Close.set_digits(2); // Number of digits in slider position
	m_Slider_Close.set_draw_value(true); // Show position label
	m_Slider_Close.set_value_pos(Gtk::POS_BOTTOM); // Where to draw the position label (if drawn at all)
	m_Slider_Close.signal_button_press_event().connect(sigc::mem_fun(*this, &guiController::on_slider_close_clicked), false);
	m_Slider_Close.signal_button_release_event().connect(sigc::mem_fun(*this, &guiController::on_slider_close_released), false);
	m_Slider_Close.signal_change_value().connect(sigc::mem_fun(*this, &guiController::on_slider_close_value_changed));
	m_Frame_Close.add(m_Slider_Close);
	m_Frame_Close.set_label("Grasp state [\"open\" ... \"close\"]");
	m_Frame_Close.set_label_align(0.02, 0.5);
	m_Frame_Close.set_border_width(5);
	m_Frame_Close.set_shadow_type(Gtk::SHADOW_NONE);

	// Velocity slider
	m_Slider_Velocity.set_digits(0); // Number of digits in slider position
	m_Slider_Velocity.set_draw_value(true); // Show position label
	m_Slider_Velocity.set_value_pos(Gtk::POS_BOTTOM); // Where to draw the position label (if drawn at all)
	m_Slider_Velocity.signal_button_press_event().connect(sigc::mem_fun(*this, &guiController::on_slider_velocity_clicked), false);
	m_Slider_Velocity.signal_button_release_event().connect(sigc::mem_fun(*this, &guiController::on_slider_velocity_released), false);
	m_Slider_Velocity.signal_change_value().connect(sigc::mem_fun(*this, &guiController::on_slider_velocity_value_changed));
	m_Frame_Velocity.add(m_Slider_Velocity);
	m_Frame_Velocity.set_label("Velocity (degrees/second)");
	m_Frame_Velocity.set_label_align(0.02, 0.5);
	m_Frame_Velocity.set_border_width(5);
	m_Frame_Velocity.set_shadow_type(Gtk::SHADOW_NONE);

	// Reactive Grasping slider
	m_Slider_Reactive.set_digits(0); // Number of digits in slider position
	m_Slider_Reactive.set_draw_value(true); // Show position label
	m_Slider_Reactive.set_value_pos(Gtk::POS_BOTTOM); // Where to draw the position label (if drawn at all)
	m_Slider_Reactive.signal_button_press_event().connect(sigc::mem_fun(*this, &guiController::on_slider_reactive_clicked), false);
	m_Slider_Reactive.signal_button_release_event().connect(sigc::mem_fun(*this, &guiController::on_slider_reactive_released), false);
	m_Slider_Reactive.signal_change_value().connect(sigc::mem_fun(*this, &guiController::on_slider_reactive_value_changed));
	m_Frame_Reactive.add(m_Slider_Reactive);
	m_Frame_Reactive.set_label("Reactive Grasp Limit");
	m_Frame_Reactive.set_label_align(0.02, 0.5);
	m_Frame_Reactive.set_border_width(5);
	m_Frame_Reactive.set_shadow_type(Gtk::SHADOW_NONE);

	// SDH assembly
	m_VBox_SDH.pack_start(m_Frame_Grasp, Gtk::PACK_EXPAND_PADDING, 0);
	m_VBox_SDH.pack_start(*Gtk::manage(new Gtk::HSeparator()), Gtk::PACK_SHRINK, 0);
	m_VBox_SDH.pack_start(m_Frame_Close, Gtk::PACK_EXPAND_PADDING, 0);
	m_VBox_SDH.pack_start(*Gtk::manage(new Gtk::HSeparator()), Gtk::PACK_SHRINK, 0);
	m_VBox_SDH.pack_start(m_Frame_Velocity, Gtk::PACK_EXPAND_PADDING, 0);
	m_VBox_SDH.pack_start(*Gtk::manage(new Gtk::HSeparator()), Gtk::PACK_SHRINK, 0);
	m_VBox_SDH.pack_start(m_Frame_Reactive, Gtk::PACK_EXPAND_PADDING, 0);
	m_Frame_SDH.add(m_VBox_SDH);


	//--------------------
	// DSA Frame
	//--------------------
	m_Frame_DSA.set_label("Digital Sensor Array");
	m_Frame_DSA.set_label_align(0.02, 0.5);
	m_Frame_DSA.set_sensitive(false);

	// Sensitivity Slider
	m_Slider_Sensitivity.set_digits(2); // Number of digits in slider position
	m_Slider_Sensitivity.set_draw_value(true); // Show position label
	m_Slider_Sensitivity.set_value_pos(Gtk::POS_BOTTOM); // Where to draw the position label (if drawn at all)
	m_Slider_Sensitivity.signal_button_press_event().connect(sigc::mem_fun(*this, &guiController::on_slider_sensitivity_clicked), false);
	m_Slider_Sensitivity.signal_button_release_event().connect(sigc::mem_fun(*this, &guiController::on_slider_sensitivity_released), false);
	m_Slider_Sensitivity.signal_change_value().connect(sigc::mem_fun(*this, &guiController::on_slider_sensitivity_value_changed));
	m_Frame_Sensitivity.add(m_Slider_Sensitivity);
	m_Frame_Sensitivity.set_label("Sensitivity");
	m_Frame_Sensitivity.set_label_align(0.02, 0.5);
	m_Frame_Sensitivity.set_border_width(5);
	m_Frame_Sensitivity.set_shadow_type(Gtk::SHADOW_NONE);
	m_sensitivity = 1.0;

	// Threshold Slider
	m_Slider_Threshold.set_digits(0); // Number of digits in slider position
	m_Slider_Threshold.set_draw_value(true); // Show position label
	m_Slider_Threshold.set_value_pos(Gtk::POS_BOTTOM); // Where to draw the position label (if drawn at all)
	m_Slider_Threshold.signal_button_press_event().connect(sigc::mem_fun(*this, &guiController::on_slider_threshold_clicked), false);
	m_Slider_Threshold.signal_button_release_event().connect(sigc::mem_fun(*this, &guiController::on_slider_threshold_released), false);
	m_Slider_Threshold.signal_change_value().connect(sigc::mem_fun(*this, &guiController::on_slider_threshold_value_changed));
	m_threshold = 150;

	// Calibrated threshold button
	m_Button_Threshold.set_label("High Sensitivity");
	m_Button_Threshold.set_size_request(-1, 36);
	m_Button_Threshold.signal_clicked().connect(sigc::mem_fun(*this, &guiController::on_button_threshold_clicked));
	m_ButtonBox_Threshold.pack_start(m_Button_Threshold, Gtk::PACK_START, 0);

	// Reset threshold
	m_Button_Threshold_Reset.set_label("Reset");
	m_Button_Threshold_Reset.set_size_request(-1, 36);
	m_Button_Threshold_Reset.signal_clicked().connect(sigc::mem_fun(*this, &guiController::on_button_threshold_reset_clicked));
	m_ButtonBox_Threshold.pack_start(m_Button_Threshold_Reset, Gtk::PACK_START, 0);

	m_VBox_Threshold.pack_start(m_Slider_Threshold, Gtk::PACK_EXPAND_PADDING, 0);
	m_VBox_Threshold.pack_start(m_ButtonBox_Threshold, Gtk::PACK_EXPAND_PADDING, 0);
	m_Frame_Threshold.add(m_VBox_Threshold);
	m_Frame_Threshold.set_label("Threshold");
	m_Frame_Threshold.set_label_align(0.02, 0.5);
	m_Frame_Threshold.set_border_width(5);
	m_Frame_Threshold.set_shadow_type(Gtk::SHADOW_NONE);

	// DSA assembly
	m_VBox_DSA.pack_start(m_Frame_Sensitivity, Gtk::PACK_EXPAND_PADDING, 0);
	m_VBox_DSA.pack_start(m_Frame_Threshold, Gtk::PACK_EXPAND_PADDING, 0);
	m_Frame_DSA.add(m_VBox_DSA);



	// Sidebar assembly
	m_VBox_Left_Sidebar.pack_start(m_Frame_Recorder_SDH, Gtk::PACK_EXPAND_PADDING, 0);
	m_VBox_Left_Sidebar.pack_start(m_Frame_Recorder_DSA, Gtk::PACK_EXPAND_PADDING, 0);
	m_VBox_Left_Sidebar.pack_start(m_Frame_SDH, Gtk::PACK_EXPAND_PADDING, 0);
	m_VBox_Left_Sidebar.pack_start(m_Frame_DSA, Gtk::PACK_EXPAND_PADDING, 0);

	add(m_VBox_Left_Sidebar);
}

guiController::~guiController() {
	// Wait for the threads to complete
	if(m_Thread_Grasp)
		m_Thread_Grasp->join();
	if(m_Thread_Grasp_Reactive)
		m_Thread_Grasp_Reactive->join();
	if(m_Thread_Grasp_Slip)
		m_Thread_Grasp_Slip->join();
}


void guiController::connectSDH() {
	if(controller->isAvailableSDH() ) {

		controller->connectSDH();

		// Start frame grabber
		frameGrabberSDH = controller->getFrameGrabberSDH();
		frameGrabberSDH->setJointAngle(m_CheckButton_JointAngles.get_active());
		frameGrabberSDH->setTemperature(m_CheckButton_Temperature.get_active());

		recorder_paused_SDH = true;
		recorder_recording_SDH = false;

		frameGrabberSDH->start(30.0, 1.0, recorder_paused_SDH, recorder_recording_SDH); // FPSJointAngles, FPSTemperature

		if(controller->isConnectedSDH()) {
			m_Frame_SDH.set_sensitive(true);
			m_Frame_Recorder_SDH.set_sensitive(true);
			m_Button_Pause_SDH.set_image(m_Image_Play_SDH);
			m_Button_Pause_SDH.set_sensitive(true);
			m_Button_Record_SDH.set_sensitive(true);
			m_Button_Stop_SDH.set_sensitive(false);
		} else {
			m_Frame_SDH.set_sensitive(false);
			m_Frame_Recorder_SDH.set_sensitive(false);
		}
	} else {
		Gtk::MessageDialog dialog("SDH not found!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
		dialog.run();
	}
}


void guiController::disconnectSDH() {
	frameGrabberSDH->finish();
	controller->disconnectSDH();
	m_Frame_SDH.set_sensitive(false);
	m_Frame_Recorder_SDH.set_sensitive(false);
}


void guiController::on_button_grasp_clicked() {
	m_Button_Grasp.set_sensitive(false);
	m_Button_Grasp_Reactive.set_sensitive(false);
	m_ToggleButton_Grasp_Slip.set_sensitive(false);

	Glib::ustring graspName = m_Combo_Grasp.get_active_text();
	int graspID = m_Combo_Grasp.get_active_row_number();
	double closeRatio = m_Adjustment_Close.get_value();
	double velocity = m_Adjustment_Velocity.get_value();

	// Create a joinable thread
	if(!m_Thread_Grasp) {
		printf("Executing grasp %d (%s): %f\n", graspID, graspName.c_str(), closeRatio);
		m_Thread_Grasp = Glib::Thread::create(
				sigc::bind<int,double,double>(
						sigc::mem_fun(*this,&guiController::worker_grasp),
						graspID, closeRatio, velocity), true);
	}
}


void guiController::on_button_grasp_reactive_clicked() {

	if(controller->isConnectedDSA()) {

		if(frameGrabberDSA->isCapturing()) {

			m_Button_Grasp.set_sensitive(false);
			m_Button_Grasp_Reactive.set_sensitive(false);
			m_ToggleButton_Grasp_Slip.set_sensitive(false);

			Glib::ustring graspName = m_Combo_Grasp.get_active_text();
			int graspID = m_Combo_Grasp.get_active_row_number();
			double velocity = m_Adjustment_Velocity.get_value();
			double limit = m_Adjustment_Reactive.get_value();

			// Create a joinable thread
			if(!m_Thread_Grasp_Reactive) {
				printf("Executing reactive grasp %d (%s): %f mV\n", graspID, graspName.c_str(), limit);
				m_Thread_Grasp_Reactive = Glib::Thread::create(
						sigc::bind<int,double,double>(
								sigc::mem_fun(*this,&guiController::worker_grasp_reactive),
								graspID, velocity, limit), true);
			}
		}
		else {
			Gtk::MessageDialog dialog("DSA frame grabber is not running!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
			dialog.run();
		}

	} else {
		Gtk::MessageDialog dialog("DSA is not connected!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
		dialog.run();
	}
}


void guiController::on_button_grasp_slip_clicked() {

	if(!m_ToggleButton_Grasp_Slip_Failed) { // Work around "double firing" when toggle button state is manually changed

		if(!m_ToggleButton_Grasp_Slip.get_active()) {
			// Terminate thread
			if(m_Thread_Grasp_Slip) {
				Glib::Mutex::Lock lock(m_mutex_thread_grasp_slip);
				m_stop_thread_grasp_slip = true;
			}
		}
		else {
			// Try to start thread
			bool failed = true;
			if(controller->isConnectedDSA()) {
				if(frameGrabberDSA->isCapturing()) {
					if(frameManager->getSlipDetectionState()) { // Slip detection enabled

						failed = false;

						m_Button_Grasp.set_sensitive(false);
						m_Button_Grasp_Reactive.set_sensitive(false);

						Glib::ustring graspName = m_Combo_Grasp.get_active_text();
						int graspID = m_Combo_Grasp.get_active_row_number();
						double velocity = m_Adjustment_Velocity.get_value();
						double limit = m_Adjustment_Reactive.get_value();

						// Create a joinable thread
						if(!m_Thread_Grasp_Slip) {
							printf("Executing reactive grasp with slip detection %d (%s): %f mV\n", graspID, graspName.c_str(), limit);
							m_stop_thread_grasp_slip = false;
							m_Thread_Grasp_Slip = Glib::Thread::create(
									sigc::bind<int,double,double,double>(
											sigc::mem_fun(*this,&guiController::worker_grasp_slip),
											graspID, velocity, limit, 2200.0, 300.0), true);
						}

					} else {
						Gtk::MessageDialog dialog("Please enable slip detection first!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
						dialog.run();
					}
				} else {
					Gtk::MessageDialog dialog("DSA frame grabber is not running!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
					dialog.run();
				}
			} else {
				Gtk::MessageDialog dialog("DSA is not connected!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
				dialog.run();
			}

			if(failed) {
				// Despite not being connected, the toggle button's state is now "pressed in"
				m_ToggleButton_Grasp_Slip_Failed = true;
				m_ToggleButton_Grasp_Slip.set_active(false); // Signal is emitted, i.e. recursive call of on_button_grasp_slip_clicked
			}
		}
	}
	else {
		m_ToggleButton_Grasp_Slip_Failed = false;
	}
}


void guiController::on_button_relax_clicked() {
	m_Button_Grasp.set_sensitive(false);
	m_Button_Grasp_Reactive.set_sensitive(false);
	m_ToggleButton_Grasp_Slip.set_sensitive(false);

	bool isGrabbing = false;
	if(frameGrabberSDH->isCapturing()) {
		isGrabbing = true;
		frameGrabberSDH->pauseBlocking();
	}

	try
	{
		// Enable all axes:
		//hand.SetAxisEnable( hand.all_axes, hand.ones_v );

		// Disable all axes:
		controller->getSDH()->SetAxisEnable( -1, false );

		if(m_Thread_Grasp) {
			m_Thread_Grasp->join(); // Wait for possibly interrupted thread to complete
			// Give hand some time to "recover" from interruption
			boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
		}

		if(m_Thread_Grasp_Reactive) {
			m_Thread_Grasp_Reactive->join(); // Wait for possibly interrupted thread to complete
			// Give hand some time to "recover" from interruption
			boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
		}

		// Resume grabbing
		if(isGrabbing) {
			frameGrabberSDH->resume();
		}

	}
	catch(...) {
		fprintf(stderr, "Unhandled Exception. Probably garbage on communication interface: Reboot hand");
	}

	m_Button_Grasp.set_sensitive(true);
	m_Button_Grasp_Reactive.set_sensitive(true);

	// Terminate thread of reactive grasp with slip detection
	if(m_Thread_Grasp_Slip) {
		Glib::Mutex::Lock lock(m_mutex_thread_grasp_slip);
		m_stop_thread_grasp_slip = true;
	}
}


void guiController::worker_grasp(int graspID, double closeRatio, double velocity) {
	try
	{
		controller->grasp(graspID, closeRatio, velocity);
	}
	catch(...) {
		fprintf(stderr, "Unhandled Exception. Probably garbage on communication interface: Reboot hand");
	}

	m_Thread_Grasp_Dispatcher();
	throw Glib::Thread::Exit();
}


void guiController::worker_grasp_reactive(int graspID, double velocity, double limit) {
	try
	{
		controller->graspReactive(graspID, velocity, limit);
	}
	catch(...) {
		fprintf(stderr, "Unhandled Exception. Probably garbage on communication interface: Reboot hand");
	}

	m_Thread_Grasp_Reactive_Dispatcher();
	throw Glib::Thread::Exit();
}


void guiController::worker_grasp_slip(int graspID, double velocity, double limitLow, double limitHigh, double stepSize) {
	try
	{
		bool finished_movement;
		double currentLimit = limitLow;

		// Execute reactive grasp
		boost::tie(finished_movement, currentLimit) = controller->graspReactive(graspID, velocity, currentLimit);

		// Give tactile sensor some time to "level out"
		boost::this_thread::sleep(boost::posix_time::milliseconds(500));

		// Set reference frame
		for(uint m = 0; m < 6; m++) {
			if(frameManager->getSlipDetectionState(m)) {
				frameManager->setSlipReferenceFrameLive(m);
			}
		}

		while(true) {
			{ // "Interruption point" (lock disappears with scope)
				Glib::Mutex::Lock lock(m_mutex_thread_grasp_slip);
				if(m_stop_thread_grasp_slip) {
					break;
				}
			}

			if(frameManager->getSlipLiveBinary()) {
				// Increase sensor limit if slip was detected and grasp again until next limit is reached
				if(currentLimit <= limitHigh) {
					currentLimit += stepSize;

					// Execute reactive grasp
					boost::tie(finished_movement, currentLimit) = controller->graspReactive(graspID, velocity, currentLimit);

					// Give tactile sensor some time to "level out"
					boost::this_thread::sleep(boost::posix_time::milliseconds(50)); // At least a single frame

					// Set reference frame again
					for(uint m = 0; m < 6; m++) {
						if(frameManager->getSlipDetectionState(m)) {
							frameManager->setSlipReferenceFrameLive(m);
						}
					}

				}
			}

			// Final grasp state reached
			if(currentLimit > limitHigh || finished_movement) {
				printf("Slip aware reactive grasping finished!\n");
				// Exit thread
				break;
			}

		}
	}
	catch(...) {
		fprintf(stderr, "Unhandled Exception. Probably garbage on communication interface: Reboot hand");
	}

	// Exit thread
	m_Thread_Grasp_Slip_Dispatcher();
	throw Glib::Thread::Exit();
}


void guiController::on_worker_grasp_done() {
	m_Button_Grasp.set_sensitive(true);
	m_Button_Grasp_Reactive.set_sensitive(true);
	m_ToggleButton_Grasp_Slip.set_sensitive(true);
	m_Thread_Grasp = NULL;
}


void guiController::on_worker_grasp_reactive_done() {
	m_Button_Grasp.set_sensitive(true);
	m_Button_Grasp_Reactive.set_sensitive(true);
	m_ToggleButton_Grasp_Slip.set_sensitive(true);
	m_Thread_Grasp_Reactive = NULL;
}


void guiController::on_worker_grasp_slip_done() {
	m_Button_Grasp.set_sensitive(true);
	m_Button_Grasp_Reactive.set_sensitive(true);
	m_Thread_Grasp_Slip = NULL;
	m_ToggleButton_Grasp_Slip.set_active(false); // Button toggle state
}


void guiController::on_combo_grasp_changed() {
	//  Glib::ustring text = m_Combo_Grasp.get_active_text();
	//  if(!(text.empty()))
	//    std::cout << "Combo changed: " << text << std::endl;
}


bool guiController::on_slider_close_value_changed(Gtk::ScrollType type, double value) {
	// std::cout << m_Adjustment_Close.get_value() << endl;
	return true;
}


// Gtk's Hscale widgets normally "jump" to a specific position with a middle-click.
// To achieve this with the left mouse button, the event is manipulated before the widgets reacts to it
bool guiController::on_slider_close_clicked(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}


// See on_slider_close_clicked()
bool guiController::on_slider_close_released(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}


bool guiController::on_slider_velocity_value_changed(Gtk::ScrollType type, double value) {
	// std::cout << m_Adjustment_Velocity.get_value() << endl;
	return true;
}

// Gtk's Hscale widgets normally "jump" to a specific position with a middle-click.
// To achieve this with the left mouse button, the event is manipulated before the widgets reacts to it
bool guiController::on_slider_velocity_clicked(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}


// See on_slider_velocity_clicked()
bool guiController::on_slider_velocity_released(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}


bool guiController::on_slider_reactive_value_changed(Gtk::ScrollType type, double value) {
	// std::cout << m_Adjustment_Reactive.get_value() << endl;
	return true;
}


// Gtk's Hscale widgets normally "jump" to a specific position with a middle-click.
// To achieve this with the left mouse button, the event is manipulated before the widgets reacts to it
bool guiController::on_slider_reactive_clicked(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}

// See on_slider_reactive_clicked()
bool guiController::on_slider_reactive_released(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}


void guiController::connectDSA() {

	if(controller->isAvailableDSA() ) {

		controller->connectDSA();

		// Start DSA frame grabber
		recorder_paused_DSA = true;
		recorder_recording_DSA = false;

		frameGrabberDSA = controller->getFrameGrabberDSA();
		frameGrabberDSA->start(100, recorder_paused_DSA, recorder_recording_DSA);
		//frameGrabberDSA->start(0.333, recorder_paused_DSA, recorder_recording_DSA);

		if(controller->isConnectedDSA()) {
			m_Frame_DSA.set_sensitive(true);
			m_Frame_Recorder_DSA.set_sensitive(true);
			m_Button_Pause_DSA.set_image(m_Image_Play_DSA);
			m_Button_Pause_DSA.set_sensitive(true);
			m_Button_Record_DSA.set_sensitive(true);
			m_Button_Stop_DSA.set_sensitive(false);

		} else {
			m_Frame_DSA.set_sensitive(false);
			m_Frame_Recorder_DSA.set_sensitive(false);
		}

	} else {
		Gtk::MessageDialog dialog("DSA not found!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
		dialog.run();
	}
}


void guiController::disconnectDSA() {
	frameGrabberDSA->finish();
	controller->disconnectDSA();
	m_Frame_DSA.set_sensitive(false);
	m_Frame_Recorder_DSA.set_sensitive(false);
}


void guiController::on_button_pause_SDH_clicked() {

	if(controller->isConnectedSDH()) {
		if(frameGrabberSDH->isCapturing()) {
			frameGrabberSDH->pause();
		} else {
			frameGrabberSDH->resume();
		}
	}

	recorder_paused_SDH = !recorder_paused_SDH;
	if(recorder_paused_SDH) {
		m_Button_Pause_SDH.set_image(m_Image_Play_SDH);
	} else {
		m_Button_Pause_SDH.set_image(m_Image_Pause_SDH);
	}
}


void guiController::on_button_stop_SDH_clicked() {
	recorder_recording_SDH = false;
	frameGrabberSDH->suspendRecording();
	m_Button_Record_SDH.set_sensitive(true);
	m_Button_Stop_SDH.set_sensitive(false);
	m_CheckButton_Temperature.set_sensitive(true);
	m_CheckButton_JointAngles.set_sensitive(true);
}


void guiController::on_button_record_SDH_clicked() {
	recorder_recording_SDH = true;
	frameGrabberSDH->enableRecording();
	m_Button_Record_SDH.set_sensitive(false);
	m_Button_Stop_SDH.set_sensitive(true);
	m_CheckButton_Temperature.set_sensitive(false);
	m_CheckButton_JointAngles.set_sensitive(false);
}

void guiController::on_checkbutton_temperature_clicked() {
	//cout << "m_CheckButton_temperature: " << m_CheckButton_Temperature.get_active() << endl;
	frameGrabberSDH->setTemperature(m_CheckButton_Temperature.get_active());
}


void guiController::on_checkbutton_joint_angles_clicked() {
	//cout << "m_CheckButton_JointAngles: " << m_CheckButton_JointAngles.get_active() << endl;
	frameGrabberSDH->setJointAngle(m_CheckButton_JointAngles.get_active());
}


void guiController::on_button_pause_DSA_clicked() {

	if(controller->isConnectedDSA()) {
		if(frameGrabberDSA->isCapturing()) {
			frameGrabberDSA->pause();
		} else {
			frameGrabberDSA->resume();
		}
	}

	recorder_paused_DSA = !recorder_paused_DSA;
	if(recorder_paused_DSA) {
		m_Button_Pause_DSA.set_image(m_Image_Play_DSA);
	} else {
		m_Button_Pause_DSA.set_image(m_Image_Pause_DSA);
	}
}

void guiController::on_button_stop_DSA_clicked() {
	recorder_recording_DSA = false;
	frameGrabberDSA->suspendRecording();
	m_Button_Record_DSA.set_sensitive(true);
	m_Button_Stop_DSA.set_sensitive(false);
}


void guiController::on_button_record_DSA_clicked() {
	recorder_recording_DSA = true;
	frameGrabberDSA->enableRecording();
	m_Button_Record_DSA.set_sensitive(false);
	m_Button_Stop_DSA.set_sensitive(true);

}


bool guiController::on_slider_sensitivity_value_changed(Gtk::ScrollType type, double value) {
	float val = static_cast<float>( m_Adjustment_Sensitivity.get_value() );
	if(!utils::almostEqual(m_sensitivity, val, 4)) {
		m_sensitivity = val;
	}
	return true;
}


bool guiController::on_slider_sensitivity_clicked(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}


bool guiController::on_slider_sensitivity_released(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}

	// Pause grabbing for a moment to prevent conflicts on the serial port
	bool isGrabbing = false;
	if(frameGrabberDSA->isCapturing()) {
		isGrabbing = true;
		frameGrabberDSA->pause();
	}

	for(uint m = 0; m < frameManager->getNumMatrices(); m++) {
		frameManager->setSensitivity(m, m_sensitivity);
	}

	// Query matrix sensitivity + threshold
	for ( uint m = 0; m < frameManager->getNumMatrices(); m++ ) {
		printf("\n\nMatrix %d:\n", m);
		cDSA::sSensitivityInfo sensitivity_info = controller->getDSA()->GetMatrixSensitivity(m);
		cout << "  sensitivity         = " << sensitivity_info.cur_sens  << "\n";
		cout << "  factory_sensitivity = " << sensitivity_info.fact_sens << "\n";
		cout << "  threshold           = " << controller->getDSA()->GetMatrixThreshold(m)   << "\n";
	}

	// Resume grabbing
	if(isGrabbing) {
		frameGrabberDSA->resume();
	}

	return false;
}


bool guiController::on_slider_threshold_value_changed(Gtk::ScrollType type, double value) {
	m_threshold = static_cast<UInt16>( m_Adjustment_Threshold.get_value() );
	return true;
}


bool guiController::on_slider_threshold_clicked(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}


bool guiController::on_slider_threshold_released(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}

	// Pause grabbing for a moment to prevent conflicts on the serial port
	bool isGrabbing = false;
	if(frameGrabberDSA->isCapturing()) {
		isGrabbing = true;
		frameGrabberDSA->pause();
	}

	// Send command to DSA
	for(uint m = 0; m < frameManager->getNumMatrices(); m++) {
		frameManager->setThreshold(m, m_threshold);
	}

	// Query matrix sensitivity + threshold
	for ( uint m = 0; m < frameManager->getNumMatrices(); m++ ) {
		printf("\n\nMatrix %d:\n", m);
		cDSA::sSensitivityInfo sensitivity_info = controller->getDSA()->GetMatrixSensitivity(m);
		cout << "  sensitivity         = " << sensitivity_info.cur_sens  << "\n";
		cout << "  factory_sensitivity = " << sensitivity_info.fact_sens << "\n";
		cout << "  threshold           = " << controller->getDSA()->GetMatrixThreshold(m)   << "\n";
	}

	// Resume grabbing
	if(isGrabbing) {
		frameGrabberDSA->resume();
	}
	return false;
}


/**
 * Measure current temperature and set sensor threshold to individually calibrated values
 * The sensitivity of all matrices is subsequently set to 1.0
 * In order to get rid of ghosting have a look at setFilterOpening() of the frame processor
 */
void guiController::on_button_threshold_clicked() {

	if(!controller->isConnectedSDH()) {
		Gtk::MessageDialog dialog("SDH has to be connected for temperature measurement", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
		dialog.run();
	} else {

		// Pause grabbing for a moment to prevent conflicts on the serial port
		bool isGrabbingDSA = false;
		if(frameGrabberDSA->isCapturing()) {
			isGrabbingDSA = true;
			frameGrabberDSA->pause();
		}
		bool isGrabbingSDH = false;
		if(frameGrabberSDH->isCapturing()) {
			isGrabbingSDH = true;
			frameGrabberSDH->pauseBlocking();
		}

		// Set thresholds
		cSDH *hand = controller->getSDH();
		// Temperatures of axis motors 1-6 are close to corresponding matrices 0-5
		std::vector<double> temperature = hand->GetTemperature(hand->all_temperature_sensors);
		int stdErrors = 3; // Change this for larger error margin
		Calibration calib = controller->getCalibration();
		for(uint m = 0; m < 6; m++) {
			TemperatureNoise& params = calib.getTemperatureNoise(m);
			double threshold = params.slope*temperature[m+1] + params.intercept + stdErrors*params.RMSE;
			frameManager->setThreshold(m, threshold);
		}

		// Sensitivity
		m_sensitivity = 1.0;
		m_Slider_Sensitivity.set_value(m_sensitivity);
		for(uint m = 0; m < 6; m++) {
			frameManager->setSensitivity(m, m_sensitivity);
		}

		// Resume grabbing
		if(isGrabbingDSA) {
			frameGrabberDSA->resume();
		}
		if(isGrabbingSDH) {
			frameGrabberSDH->resume();
		}
	}
}


void guiController::on_button_threshold_reset_clicked() {

	// Pause grabbing for a moment to prevent conflicts on the serial port
	bool isGrabbingDSA = false;
	if(frameGrabberDSA->isCapturing()) {
		isGrabbingDSA = true;
		frameGrabberDSA->pause();
	}

	// Set thresholds back to factory defaults
	m_Slider_Threshold.set_value(150);
	for(uint m = 0; m < 6; m++) {
		double threshold = (m % 2 == 0) ? 150.0 : 300.0;
		frameManager->setThreshold(m, threshold);
		printf("Matrix: %d, Threshold: %f\n",m, threshold );
	}

	// Sensitivity
	m_sensitivity = 0.5;
	m_Slider_Sensitivity.set_value(m_sensitivity);
	for(uint m = 0; m < 6; m++) {
		frameManager->setSensitivity(m, m_sensitivity);
	}

	// Resume grabbing
	if(isGrabbingDSA) {
		frameGrabberDSA->resume();
	}
}
