#include <sstream>

#include "guiSlipDetection.h"
#include "guiMain.h"

guiSlipDetection::guiSlipDetection(Controller *c, guiMain *gui)
:
m_Adjustment_Threshold_Reference(1.0, 0.0, 25.0, 0.25, 0.0, 0.0),
m_Slider_Threshold_Reference(m_Adjustment_Threshold_Reference),
m_Adjustment_Threshold_Consecutive(0.20, 0.0, 2.0, 0.01, 0.0, 0.0),
m_Slider_Threshold_Consecutive(m_Adjustment_Threshold_Consecutive),
m_table(2, 3, true) // rows, cols, homogeneous spacing
{
	controller = c;
	frameManager = controller->getFrameManager();
	frameProcessor = frameManager->getFrameProcessor();
	mainGUI = gui;

	set_title("Slip Detection");
	set_skip_taskbar_hint(true); // No task bar entry
	set_type_hint(Gdk::WINDOW_TYPE_HINT_DIALOG); // Always on top
	set_size_request(400, 520);

	if(frameManager->isConnectedDSA()) {
		m_mode = ONLINE;
	} else if(frameManager->getTSFrameAvailable()) {
		m_mode = OFFLINE;
	} else {
		m_mode = INACTIVE;
	}

	std::ostringstream oss;

	m_startFrame.resize(6, 0);
	m_stopFrame.resize(6, 0);

	m_threshSlipReference = m_Adjustment_Threshold_Reference.get_value();
	m_threshSlipConsecutive = m_Adjustment_Threshold_Consecutive.get_value();
	m_threshSlipAngle = 5.0;

	// Records
	for(uint m = 0; m < 6; m++) {
		m_slipResults.push_back( boost::make_shared<std::deque<slipResult> > () );
		m_slipvectors.push_back( boost::make_shared<std::deque<slip_trajectory> > () );
		m_slipangles.push_back( boost::make_shared<std::deque<double> > () );
	}

	// Images
	m_Image_grey.set("icons/blinkenlight_grey.svg");
	m_Image_green.set("icons/blinkenlight_green.svg");
	m_Image_amber.set("icons/blinkenlight_amber.svg");
	m_Image_red.set("icons/blinkenlight_red.svg");

	m_blinkenlight_actual.resize(6, GREY);
	m_blinkenlight_target.resize(6, GREY);

	// Slip threshold reference
	m_Slider_Threshold_Reference.set_digits(2); // Number of digits in slider position
	m_Slider_Threshold_Reference.set_draw_value(true); // Show position label
	m_Slider_Threshold_Reference.set_value_pos(Gtk::POS_BOTTOM); // Where to draw the position label (if drawn at all)
	m_Slider_Threshold_Reference.signal_button_press_event().connect(sigc::mem_fun(*this, &guiSlipDetection::on_slider_threshold_reference_clicked), false);
	m_Slider_Threshold_Reference.signal_button_release_event().connect(sigc::mem_fun(*this, &guiSlipDetection::on_slider_threshold_reference_released), false);
	m_Slider_Threshold_Reference.signal_change_value().connect(sigc::mem_fun(*this, &guiSlipDetection::on_slider_threshold_reference_value_changed));
	m_Frame_Threshold_Reference.add(m_Slider_Threshold_Reference);
	m_Frame_Threshold_Reference.set_label("Slip-state reference");
	m_Frame_Threshold_Reference.set_label_align(0.02, 0.5);
	m_Frame_Threshold_Reference.set_border_width(5);
	m_Frame_Threshold_Reference.set_shadow_type(Gtk::SHADOW_NONE);

	// Slip threshold consecutive
	m_Slider_Threshold_Consecutive.set_digits(2); // Number of digits in slider position
	m_Slider_Threshold_Consecutive.set_draw_value(true); // Show position label
	m_Slider_Threshold_Consecutive.set_value_pos(Gtk::POS_BOTTOM); // Where to draw the position label (if drawn at all)
	m_Slider_Threshold_Consecutive.signal_button_press_event().connect(sigc::mem_fun(*this, &guiSlipDetection::on_slider_threshold_consecutive_clicked), false);
	m_Slider_Threshold_Consecutive.signal_button_release_event().connect(sigc::mem_fun(*this, &guiSlipDetection::on_slider_threshold_consecutive_released), false);
	m_Slider_Threshold_Consecutive.signal_change_value().connect(sigc::mem_fun(*this, &guiSlipDetection::on_slider_threshold_consecutive_value_changed));
	m_Frame_Threshold_Consecutive.add(m_Slider_Threshold_Consecutive);
	m_Frame_Threshold_Consecutive.set_label("Slip-state consecutive");
	m_Frame_Threshold_Consecutive.set_label_align(0.02, 0.5);
	m_Frame_Threshold_Consecutive.set_border_width(5);
	m_Frame_Threshold_Consecutive.set_shadow_type(Gtk::SHADOW_NONE);


	// Create widgets for each matrix
	for(uint m = 0; m < 6; m++) {

		// Checkbutton
		Gtk::CheckButton* checkbutton = Gtk::manage( new Gtk::CheckButton("Enable") );
		checkbutton->signal_clicked().connect(sigc::bind(sigc::mem_fun(*this, &guiSlipDetection::on_checkbutton_enable_clicked), m));
		if(m_mode == OFFLINE || m_mode == INACTIVE) checkbutton->set_sensitive(false);
		m_checkbuttons.push_back(checkbutton);

		// Button
		Gtk::Button* button = Gtk::manage( new Gtk::Button() );
		button->signal_clicked().connect(sigc::bind(sigc::mem_fun(*this, &guiSlipDetection::on_button_set_reference_clicked), m));
		if(m_mode == ONLINE) button->set_label("Set reference");
		if(m_mode == OFFLINE) button->set_label("Compute slip");
		if(m_mode == INACTIVE) {
			button->set_label("");
			button->set_sensitive(false);
		}
		m_buttons.push_back(button);

		// ToggleButton
		Gtk::ToggleButton* togglebutton = Gtk::manage( new Gtk::ToggleButton("Show Graph") );
		togglebutton->signal_clicked().connect(sigc::bind(sigc::mem_fun(*this, &guiSlipDetection::on_togglebutton_details_clicked), m));
		if(m_mode == INACTIVE) togglebutton->set_sensitive(false);
		m_togglebuttons.push_back(togglebutton);

		// Status Label
		Gtk::Label* label = Gtk::manage( new Gtk::Label("Slip:") );
		m_statuslabels.push_back(label);

		// Status Image
		Gtk::Image* image = Gtk::manage( new Gtk::Image() );
		image->set(m_Image_grey.get_pixbuf());
		m_statusimages.push_back(image);

		// Status HBox
		Gtk::HBox* hbox = Gtk::manage( new Gtk::HBox() );
		hbox->pack_start(*label, Gtk::PACK_SHRINK, 5);
		hbox->pack_start(*image, Gtk::PACK_SHRINK, 5);
		m_statushboxes.push_back(hbox);

		// Pack all widgets in VBox
		Gtk::VBox* vbox = Gtk::manage(new Gtk::VBox());
		vbox->pack_start(*checkbutton, Gtk::PACK_SHRINK, 5);
		vbox->pack_start(*button, Gtk::PACK_SHRINK, 5);
		vbox->pack_start(*togglebutton, Gtk::PACK_SHRINK, 5);
		vbox->pack_start(*hbox, Gtk::PACK_SHRINK, 5);
		vbox->set_border_width(5);
		m_vboxes.push_back(vbox);

		//Add VBox to Frame
		oss.str(""); // clear
		oss << "Matrix " << m;
		Gtk::Frame* frame = Gtk::manage( new Gtk::Frame(oss.str()) );
		frame->add(*vbox);
		frame->set_border_width(5);
		m_frames.push_back(frame);

		m_multiplots.push_back(NULL); // Initialized later
	}

	m_table.attach(*m_frames[1], 0, 1, 0, 1); // (0,1)
	m_table.attach(*m_frames[3], 1, 2, 0, 1); // (0,2)
	m_table.attach(*m_frames[5], 2, 3, 0, 1); // (0,3)
	m_table.attach(*m_frames[0], 0, 1, 1, 2); // (1,1)
	m_table.attach(*m_frames[2], 1, 2, 1, 2); // (1,2)
	m_table.attach(*m_frames[4], 2, 3, 1, 2); // (1,3)
	m_table.set_border_width(2);

	m_VBox_Main.pack_start(m_table, Gtk::PACK_START, 0);
	m_VBox_Main.pack_start(m_Frame_Threshold_Reference, Gtk::PACK_START, 0);
	m_VBox_Main.pack_start(m_Frame_Threshold_Consecutive, Gtk::PACK_START, 0);


	add(m_VBox_Main);

	show_all();
}


guiSlipDetection::~guiSlipDetection() {
	for(uint m = 0; m < 6; m++) {
		delete m_multiplots[m];
	}
}


void guiSlipDetection::clearTrajectory(uint m) {
	m_slipResults[m]->clear();
	m_slipvectors[m]->clear();
	m_slipangles[m]->clear();
}


void guiSlipDetection::setModeOnline() {
	m_mode = ONLINE;
	for(uint m = 0; m < 6; m++ ) {
		m_checkbuttons[m]->set_sensitive(true);
		m_checkbuttons[m]->set_active(false);
		m_buttons[m]->set_label("Set reference");
		m_buttons[m]->set_sensitive(true);
		m_togglebuttons[m]->set_sensitive(true);
		clearTrajectory(m);
		if(m_multiplots[m] != NULL) {
			m_multiplots[m]->setAxisLimits(0, 50, -50, 50);
			m_multiplots[m]->reset();
		}
	}
}


void guiSlipDetection::setModeOffline() {
	m_mode = OFFLINE;
	if(m_connection.connected()) {
		m_connection.disconnect();
	}
	for(uint m = 0; m < 6; m++ ) {
		m_checkbuttons[m]->set_sensitive(false);
		m_checkbuttons[m]->set_active(false);
		m_buttons[m]->set_label("Compute slip");
		m_buttons[m]->set_sensitive(true);
		m_togglebuttons[m]->set_sensitive(true);
		frameManager->disableSlipDetection(m);
		clearTrajectory(m);
	}
}


bool guiSlipDetection::runSlipDetectionOnline() {

	// Since all GUI elements run in a single Gtk loop,
	// only enter "blocked waiting" if frame grabber is actually running.
	// A deadlock would be the result otherwise.
	if(controller->isConnectedDSA()) {
		if(controller->getFrameGrabberDSA()->isCapturing()) {

			std::vector<boost::optional<slipResult> > slipResults;

			// Fetch slip detection result from queue
			slipResults = frameManager->slipResultConsumer();

			for(uint m = 0; m < 6; m++) {
				if(slipResults[m]) {

					// Build cumulative slip trajectory
					double dx, dy, slipangle;
					if(m_slipvectors[m]->empty()) { // First measurement
						dx = 0.0;
						dy = 0.0;
						slipangle = 0.0;
					} else {
						boost::tie(dx, dy) = m_slipvectors[m]->back();
						slipangle = m_slipangles[m]->back();
					}

					slipResult &slip = *(slipResults[m]);
					m_slipResults[m]->push_back(slip);
					m_slipvectors[m]->push_back( boost::make_tuple(dx+slip.slipVector_x, dy+slip.slipVector_y) );
					m_slipangles[m]->push_back(slipangle+slip.slipAngle);

					drawTrajectoryOnline(m);
					//printf("Matrix: %d:  %2.3f, %2.3f\n", m, slip.slipVector_x, slip.slipVector_y);

					// Blinkenlights
					if(slip.successTranslation) {

						bool slipStateReference = fabs(slip.slipVectorReference_x) > m_threshSlipReference ||
								fabs(slip.slipVectorReference_y) > m_threshSlipReference;

						bool slipStateConsecutive = fabs(slip.slipVector_x) > m_threshSlipConsecutive ||
								fabs(slip.slipVector_y) > m_threshSlipConsecutive;

						if(slipStateReference || slipStateConsecutive) {
							m_blinkenlight_target[m] = RED; // :-(
						} else {
							m_blinkenlight_target[m] = GREEN; // :-)
						}
					} else {
						m_blinkenlight_target[m] = AMBER; // :-|
					}
				}
			}
		} else { // Not capturing
			for(uint m = 0; m < 6; m++) {
				m_blinkenlight_target[m] = GREY;
			}
		}
	} else { // Not connected
		for(uint m = 0; m < 6; m++) {
			m_blinkenlight_target[m] = GREY;
		}
	}

	setBlinkenLights();

	return true; // true: emit again, false: stop signal
}


bool guiSlipDetection::drawTrajectoryOnline(uint m) {
	if(m_multiplots[m] != NULL) {
		slipResult& slip = m_slipResults[m]->back();
		std::deque<slip_trajectory>& slipvectors = *(m_slipvectors[m].get());
		std::deque<double>& slipangles = *(m_slipangles[m].get());
		m_multiplots[m]->updateTrajectory(slip, slipvectors, slipangles, slipvectors.size()-1 );
	}
	return false; // true: emit again, false: stop signal
}


void guiSlipDetection::setCurrentFrameOffline(uint frameID) {
	for(uint m = 0; m < 6; m++ ) {
		drawTrajectoryOffline(m, frameID);
	}
}


// Set reference frame and compute consecutive slip for entire profile
void guiSlipDetection::runSlipDetectionOffline(uint m, uint startFrame, uint stopFrame) {
	if(m_multiplots[m] != NULL) {
		m_multiplots[m]->reset(); // Set widgets to initial state before manipulating underlying data
	}

	frameManager->setSlipReferenceFrame(m, startFrame);

	// Delete outdated results and compute new slip trajectory
	clearTrajectory(m);
	m_slip = frameManager->computeSlip(m, startFrame);
	m_slipResults[m]->push_back(m_slip);
	m_slipvectors[m]->push_back( boost::make_tuple(0.0, 0.0) );
	m_slipangles[m]->push_back(0.0);

	for(uint frameID = startFrame+1; frameID <= stopFrame; frameID++) {

		// Compute slip
		m_slip = frameManager->computeSlip(m, frameID);

		// Build cumulative slip trajectory
		double dx, dy;
		boost::tie(dx, dy) = m_slipvectors[m]->back();
		double slipangle = m_slipangles[m]->back();
		m_slipResults[m]->push_back(m_slip);
		m_slipvectors[m]->push_back( boost::make_tuple(dx+m_slip.slipVector_x, dy+m_slip.slipVector_y) );
		m_slipangles[m]->push_back(slipangle+m_slip.slipAngle);
	}

	// Compute plot axis limit since all values are already known
	if(m_multiplots[m] != NULL) {
		double minAngle = *min_element(m_slipangles[m]->begin(), m_slipangles[m]->end());
		double maxAngle = *max_element(m_slipangles[m]->begin(), m_slipangles[m]->end());
		m_multiplots[m]->setAxisLimits(0, m_slipangles[m]->size()+1, minAngle, maxAngle);
	}

	drawTrajectoryOffline(m, stopFrame);

}


bool guiSlipDetection::drawTrajectoryOffline(uint m, uint currentFrameID) {

	// Clamp current frame to selected range
	if(currentFrameID < m_startFrame[m])
		currentFrameID = m_startFrame[m];
	else if(currentFrameID > m_stopFrame[m])
		currentFrameID = m_stopFrame[m];
	uint stopFrame = currentFrameID - m_startFrame[m];

	if(m_multiplots[m] != NULL) {
		if(stopFrame != 0) {
			slipResult& slip = m_slipResults[m]->at(stopFrame);
			std::deque<slip_trajectory>& slipvectors = *(m_slipvectors[m].get());
			std::deque<double>& slipangles = *(m_slipangles[m].get());
			m_multiplots[m]->drawTrajectory(slip, slipvectors, slipangles, stopFrame);
		}
	}

	// Blinkenlights
	if(stopFrame != 0) {
		slipResult& slip = m_slipResults[m]->at(stopFrame);
		if(slip.successTranslation) {

			bool slipStateReference = fabs(slip.slipVectorReference_x) > m_threshSlipReference ||
					fabs(slip.slipVectorReference_y) > m_threshSlipReference;

			bool slipStateConsecutive = fabs(slip.slipVector_x) > m_threshSlipConsecutive ||
					fabs(slip.slipVector_y) > m_threshSlipConsecutive;

			if(slipStateReference || slipStateConsecutive) {
				m_blinkenlight_target[m] = RED; // :-(
			} else {
				m_blinkenlight_target[m] = GREEN; // :-)
			}
		} else {
			m_blinkenlight_target[m] = AMBER; // :-|
		}
	} else {
		for(uint m = 0; m < 6; m++) {
			m_blinkenlight_target[m] = GREY;
		}
	}
	setBlinkenLights();
	return false; // true: emit again, false: stop signal
}


// Update bitmap only if something changed
void guiSlipDetection::setBlinkenLights() {
	for(uint m = 0; m < 6; m++) {
		if(m_blinkenlight_actual[m] != m_blinkenlight_target[m]) {

			switch(m_blinkenlight_target[m]) {
			case GREY:
				m_statusimages[m]->set(m_Image_grey.get_pixbuf());
				m_blinkenlight_actual[m] = GREY;
				break;
			case GREEN:
				m_statusimages[m]->set(m_Image_green.get_pixbuf());
				m_blinkenlight_actual[m] = GREEN;
				break;
			case AMBER:
				m_statusimages[m]->set(m_Image_amber.get_pixbuf());
				m_blinkenlight_actual[m] = AMBER;
				break;
			case RED:
				m_statusimages[m]->set(m_Image_red.get_pixbuf());
				m_blinkenlight_actual[m] = RED;
				break;
			default:
				throw std::runtime_error("unreachable");
			}
		}
	}
}


void guiSlipDetection::on_checkbutton_enable_clicked(uint m) {

	if(m_checkbuttons[m]->get_active()) { // Enable slip detection
		if(m_mode == OFFLINE) {
			if(mainGUI->getActiveSelection()) {
				m_startFrame[m] = mainGUI->getSelectionFrom();
				m_stopFrame[m] =  mainGUI->getSelectionTo();
				runSlipDetectionOffline(m, m_startFrame[m], m_stopFrame[m]);
			} else {
				m_checkbuttons[m]->set_active(false);
				Gtk::MessageDialog dialog("Selection is empty!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
				dialog.run();
			}
		}
		else if(m_mode == ONLINE) {
			frameManager->enableSlipDetection(m);
			if (!m_connection.connected()) {
				m_connection = Glib::signal_idle().connect(sigc::mem_fun(*this, &guiSlipDetection::runSlipDetectionOnline), Glib::PRIORITY_LOW);
			}
		}

	} else { // Disable slip detection
		if(m_mode == ONLINE) {

			if(m_connection.connected()) {
				m_connection.disconnect();
			}
			frameManager->disableSlipDetection(m);
			for(uint m = 0; m < 6; m++) {
				m_blinkenlight_target[m] = GREY;
			}
			setBlinkenLights();
		}
	}
}


void guiSlipDetection::on_button_set_reference_clicked(uint m) {
	if(m_mode == OFFLINE) {
		if(mainGUI->getActiveSelection()) {
			m_startFrame[m] = mainGUI->getSelectionFrom();
			m_stopFrame[m] =  mainGUI->getSelectionTo();
			printf("Setting slip interval: %d, %d\n", m_startFrame[m], m_stopFrame[m]);
			runSlipDetectionOffline(m, m_startFrame[m], m_stopFrame[m]);
		} else {
			Gtk::MessageDialog dialog("Selection is empty!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
			dialog.run();
		}
	}
	else if(m_mode == ONLINE) {
		frameManager->setSlipReferenceFrameLive(m);
		clearTrajectory(m);
		if(m_multiplots[m] != NULL) {
			m_multiplots[m]->setAxisLimits(0, 50, -50, 50);
			m_multiplots[m]->reset();
		}
	}
}

void guiSlipDetection::on_togglebutton_details_clicked(uint m) {
	if(m_multiplots[m] == NULL) { // First time: create window
		m_multiplots[m] = new guiSlipDetectionMultiPlot(); // Gtk::manage cannot be used for Gtk::Window
		m_multiplots[m]->signal_delete_event().connect(sigc::bind(sigc::mem_fun(*this, &guiSlipDetection::on_delete_detail_clicked), m));
		if(m_mode == OFFLINE) {
			// Give GTK some time to show window before drawing
			Glib::signal_timeout().connect( sigc::bind<uint>(sigc::mem_fun(*this, &guiSlipDetection::drawTrajectoryOffline), m, m_stopFrame[m]), 100);
		}
		else if(m_mode == ONLINE) {
			m_multiplots[m]->setAxisLimits(0, 50, -50, 50);
		}

	} else {
		if(!m_multiplots[m]->get_visible()) {
			m_multiplots[m]->set_visible(true); // Show window
			if(m_mode == OFFLINE) {
				drawTrajectoryOffline(m, m_stopFrame[m]);
			}
		} else {
			m_multiplots[m]->set_visible(false); // Hide window
		}
	}
}


bool guiSlipDetection::on_delete_detail_clicked(GdkEventAny* event, uint m) {
	m_togglebuttons[m]->set_active(false);
	return true;
}


bool guiSlipDetection::on_slider_threshold_reference_value_changed(Gtk::ScrollType type, double value) {
	m_threshSlipReference = m_Adjustment_Threshold_Reference.get_value();
	frameManager->setSlipThresholdReference(m_threshSlipReference);
	return true;
}

// Gtk's Hscale widgets normally "jump" to a specific position with a middle-click.
// To achieve this with the left mouse button, the event is manipulated before the widgets reacts to it
bool guiSlipDetection::on_slider_threshold_reference_clicked(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}


bool guiSlipDetection::on_slider_threshold_reference_released(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}


bool guiSlipDetection::on_slider_threshold_consecutive_value_changed(Gtk::ScrollType type, double value) {
	m_threshSlipConsecutive = m_Adjustment_Threshold_Consecutive.get_value();
	frameManager->setSlipThresholdConsecutive(m_threshSlipConsecutive);
	return true;
}


// Gtk's Hscale widgets normally "jump" to a specific position with a middle-click.
// To achieve this with the left mouse button, the event is manipulated before the widgets reacts to it
bool guiSlipDetection::on_slider_threshold_consecutive_clicked(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}


bool guiSlipDetection::on_slider_threshold_consecutive_released(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}
