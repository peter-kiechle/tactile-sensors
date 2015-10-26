
#include "guiSeekbar.h"
#include "controller.h"
#include "guiMain.h"
#include "utils.h"

guiSeekbar::guiSeekbar(Controller *c, guiMain *gui)
:
replaying(false),
m_HBox_Seekbar(false, 0),
m_Adjustment_Seekbar(0.0, 0.0, 1.0, 1.0, 0.0, 0.0), //initial value, lower, upper, step_increment, page_increment(pointless), page_size(pointless), max value = upper - page_size
m_Slider_Seekbar(m_Adjustment_Seekbar),
m_Label_Seekbar(" / 0")
{
	controller = c;
	frameManager = controller->getFrameManager();
	mainGUI = gui;

	set_shadow_type(Gtk::SHADOW_NONE);

	// Video position slider:
	m_Slider_Seekbar.set_digits(0); //number of digits in slider position
	m_Slider_Seekbar.set_draw_value(true); //don't show position label
	m_Slider_Seekbar.set_value_pos(Gtk::POS_RIGHT); //where to draw the position label (if drawn at all)
	m_Slider_Seekbar.signal_button_press_event().connect(sigc::mem_fun(*this, &guiSeekbar::on_slider_clicked), false);
	m_Slider_Seekbar.signal_button_release_event().connect(sigc::mem_fun(*this, &guiSeekbar::on_slider_released), false);
	m_Slider_Seekbar.signal_change_value().connect(sigc::mem_fun(*this, &guiSeekbar::on_slider_value_changed));
	m_Slider_Seekbar.set_sensitive(false);

	// Label
	m_Label_Seekbar.set_size_request(100, 30);
	m_Label_Seekbar.set_alignment(Gtk::ALIGN_LEFT, Gtk::ALIGN_CENTER);
	m_Label_Seekbar.set_sensitive(false);

	m_HBox_Seekbar.pack_start(m_Slider_Seekbar, Gtk::PACK_EXPAND_WIDGET);
	m_HBox_Seekbar.pack_start(m_Label_Seekbar, Gtk::PACK_SHRINK);

	// Buttons
	m_Button_Play.set_use_stock(true);
	replaying ? m_Button_Play.set_label(Gtk::Stock::MEDIA_PAUSE.id) : m_Button_Play.set_label(Gtk::Stock::MEDIA_PLAY.id);
	m_Button_Play.set_size_request(120, 30);
	m_Button_Play.signal_clicked().connect(sigc::mem_fun(*this, &guiSeekbar::on_button_play_clicked));
	m_Button_Play.set_sensitive(false);

	m_Button_Next.set_use_stock(true);
	m_Button_Next.set_label(Gtk::Stock::MEDIA_NEXT.id);
	m_Button_Next.signal_clicked().connect(sigc::mem_fun(*this, &guiSeekbar::on_button_next_clicked));
	m_Button_Next.set_sensitive(false);

	m_Button_Prev.set_use_stock(true);
	m_Button_Prev.set_label(Gtk::Stock::MEDIA_PREVIOUS.id);
	m_Button_Prev.signal_clicked().connect(sigc::mem_fun(*this, &guiSeekbar::on_button_prev_clicked));
	m_Button_Prev.set_sensitive(false);

	// Separator
	//	m_Separator_Seekbar.set_size_request(20,5);

	m_HBox_Controls.pack_start(m_Button_Play, Gtk::PACK_SHRINK);
	m_HBox_Controls.pack_start( *Gtk::manage(new Gtk::VSeparator()), Gtk::PACK_SHRINK, 5);
	m_HBox_Controls.pack_start(m_Button_Prev, Gtk::PACK_SHRINK);
	m_HBox_Controls.pack_start( *Gtk::manage(new Gtk::VSeparator()), Gtk::PACK_SHRINK, 5);
	m_HBox_Controls.pack_start(m_Button_Next, Gtk::PACK_SHRINK);

	m_VBox_Combined.pack_start(m_HBox_Seekbar,  Gtk::PACK_SHRINK, 0);
	m_VBox_Combined.pack_start(m_HBox_Controls,  Gtk::PACK_SHRINK, 0);

	add(m_VBox_Combined);
}


guiSeekbar::~guiSeekbar() { }


void guiSeekbar::initSeekbar() {
	// Adjust the slider properties (one tick per frame)
	m_Adjustment_Seekbar.set_value(1.0); //initial value
	m_Adjustment_Seekbar.set_lower(1.0);
	m_Adjustment_Seekbar.set_upper(static_cast<double>(controller->getFrameManager()->getFrameCountTS())); // max value = upper - page_size
	m_Adjustment_Seekbar.set_step_increment(1.0);
	m_Adjustment_Seekbar.set_page_increment(0.0);
	m_Adjustment_Seekbar.set_page_size(0.0);
	m_Slider_Seekbar.set_adjustment(m_Adjustment_Seekbar);

	// Update label
	std::ostringstream buffer;
	buffer << " / " << controller->getFrameManager()->getFrameCountTS();
	m_Label_Seekbar.set_label(buffer.str());

	m_Slider_Seekbar.set_sensitive(true);
	m_Label_Seekbar.set_sensitive(true);
	m_Button_Play.set_sensitive(true);
	m_Button_Prev.set_sensitive(true);
	m_Button_Next.set_sensitive(true);
}


void guiSeekbar::resetSeekbar() {
	// Adjust the slider properties (one tick per frame)
	m_Adjustment_Seekbar.set_value(0.0); //initial value
	m_Adjustment_Seekbar.set_lower(0.0);
	m_Adjustment_Seekbar.set_upper(1.0); // max value = upper - page_size
	m_Adjustment_Seekbar.set_step_increment(1.0);
	m_Adjustment_Seekbar.set_page_increment(0.0);
	m_Adjustment_Seekbar.set_page_size(0.0);
	m_Slider_Seekbar.set_adjustment(m_Adjustment_Seekbar);

	// Update label
	m_Label_Seekbar.set_label(" / 0");

	m_Slider_Seekbar.set_sensitive(false);
	m_Label_Seekbar.set_sensitive(false);
	m_Button_Play.set_sensitive(false);
	m_Button_Prev.set_sensitive(false);
	m_Button_Next.set_sensitive(false);
}


void guiSeekbar::on_button_play_clicked() {

	if(!replaying) { // Start replay
		// Get current system time and the current frame's time stamp in milliseconds
		t0 = utils::getCurrentTimeMilliseconds();
		timestamp0 = frameManager->getCurrentFrame()->timestamp;

		// Let GTKMM refresh the sensor frame in accordance with recorded frame rate
		//connection = Glib::signal_idle().connect(sigc::mem_fun(*this, &guiSeekbar::on_idle), Glib::PRIORITY_DEFAULT_IDLE);
		sigc::slot<bool> slot = sigc::mem_fun(*this, &guiSeekbar::on_signal_timeout);
		connection = Glib::signal_timeout().connect(slot, 0, Glib::PRIORITY_DEFAULT); // Note: Since on_signal_timeout() returns true, this function is called only once

		m_Button_Play.set_label(Gtk::Stock::MEDIA_PAUSE.id);
		replaying = true;

	} else { // Pause replay
		// Diconnect the signal handler:
		connection.disconnect();
		m_Button_Play.set_label(Gtk::Stock::MEDIA_PLAY.id);
		replaying = false;
	}
}


void guiSeekbar::on_button_prev_clicked() {
	int prevFrameID = frameManager->getCurrentFrameID()-1;
	if(prevFrameID > 0) {
		mainGUI->setCurrentFrame(prevFrameID);
	}
}


void guiSeekbar::on_button_next_clicked() {
	uint nextFrameID = frameManager->getCurrentFrameID()+1;
	if(nextFrameID < frameManager->getFrameCountTS()) { // Still frames remaining
		mainGUI->setCurrentFrame(nextFrameID);
	}
}


/**
 * this timer callback function is called every 1/fps seconds
 * and updates the the current video frame (Gtk::Image) periodically
 */
bool guiSeekbar::on_signal_timeout() {

	uint64_t currentTime = utils::getCurrentTimeMilliseconds(); // get current system time in milliseconds
	uint currentFrameID = frameManager->getCurrentFrameID();
	uint64_t currentTimestamp = frameManager->getFrame(currentFrameID)->timestamp;

	uint nextFrameID = currentFrameID + 1;

	if(nextFrameID < frameManager->getFrameCountTS()) { // Still frames remaining

		uint64_t nextTimestamp = frameManager->getFrame(nextFrameID)->timestamp;
		uint64_t timeToNextFrame = nextTimestamp - currentTimestamp; // Time between the current and next frame

		uint64_t actualElapsedTime = currentTime - t0;
		uint64_t targetElapsedTime = currentTimestamp - timestamp0;

		int delay = (targetElapsedTime - actualElapsedTime) + timeToNextFrame;
		if(delay < 0) { // Too slow: skip frame if possible
			if(nextFrameID+1 < frameManager->getFrameCountTS()) {
				nextFrameID++;
			}
			delay = 0;
		}

		// Add timeout for next frame
		sigc::slot<bool> slot = sigc::mem_fun(*this, &guiSeekbar::on_signal_timeout);
		connection = Glib::signal_timeout().connect(slot, delay, Glib::PRIORITY_DEFAULT); // Note: Since on_signal_timeout() returns false, this function is called only once

		mainGUI->setCurrentFrame(nextFrameID);

	} else { // Reached end
		connection.disconnect(); // Disconnect the signal handler:
		m_Button_Play.set_label(Gtk::Stock::MEDIA_PLAY.id);
		replaying = false;
	}

	return false;
}


/**
 * Update navigation bar
 */
void guiSeekbar::setSliderPosition(int frameID) {
	m_Slider_Seekbar.set_value(frameID+1);
}


bool guiSeekbar::on_slider_value_changed(Gtk::ScrollType type, double value) {
	if(buttonClicked) { // Only fire on click event (not on release)
		if(!replaying) {
			mainGUI->setCurrentFrame(m_Adjustment_Seekbar.get_value()-1);
		} else { // Pause replay

			// Diconnect the signal handler:
			connection.disconnect();

			mainGUI->setCurrentFrame(m_Adjustment_Seekbar.get_value()-1);

			// Get current system time and the current frame's time stamp in milliseconds
			t0 = utils::getCurrentTimeMilliseconds();
			timestamp0 = frameManager->getCurrentFrame()->timestamp;

			// Let GTKMM refresh the sensor frame in accordance with recorded frame rate
			//connection = Glib::signal_idle().connect(sigc::mem_fun(*this, &guiSeekbar::on_idle), Glib::PRIORITY_DEFAULT_IDLE);
			sigc::slot<bool> slot = sigc::mem_fun(*this, &guiSeekbar::on_signal_timeout);
			connection = Glib::signal_timeout().connect(slot, 0, Glib::PRIORITY_DEFAULT); // Note: Since on_signal_timeout() returns true, this function is called only once
		}
	}
	return true;
}


// Gtk's Hscale widgets normally "jump" to a specific position with a middle-click.
// To achieve this with the left mouse button, the event is manipulated before the widgets reacts to it
bool guiSeekbar::on_slider_clicked(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	buttonClicked = true;
	return false;
}


// See on_slider_clicked()
bool guiSeekbar::on_slider_released(GdkEventButton* event) {
	if(event->button == 1) { // left click
		event->button = 2; // middle click
	}
	buttonClicked = false;
	return false;
}

