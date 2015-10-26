#ifndef GUISEEKBAR_H_
#define GUISEEKBAR_H_

#include <stdint.h>
#include <gtkmm.h>


class Controller;
class FrameManager;
class guiMain;

/**
 * @class guiSeekbar
 * @brief The seekbar.
 */
class guiSeekbar : public Gtk::Frame  {

public:
	guiSeekbar(Controller *c, guiMain *gui);
	virtual ~guiSeekbar();

	void initSeekbar();
	void resetSeekbar();
	void setSliderPosition(int frameID);

private:
	Controller *controller;
	FrameManager *frameManager;
	guiMain* mainGUI;

	bool connected;
	bool replaying;
	bool buttonClicked;

	// Replaying
	uint64_t t0; // Initial time of replay
	uint64_t timestamp0; // Initial time stamp

	Gtk::VBox m_VBox_Combined;
	Gtk::HBox m_HBox_Seekbar;
	Gtk::HBox m_HBox_Controls;
	Gtk::Button m_Button_Play;
	Gtk::Button m_Button_Prev;
	Gtk::Button m_Button_Next;
	Gtk::VSeparator m_Separator_Seekbar;
	Gtk::Adjustment m_Adjustment_Seekbar;
	Gtk::HScale m_Slider_Seekbar;
	Gtk::Label m_Label_Seekbar;

	sigc::connection connection; // timeout signal

protected:

	bool on_idle(); //  idle signal handler - called as quickly as possible
	bool on_signal_timeout(); // timeout signal handler

	// Signal handlers:
	void on_button_play_clicked();
	void on_button_next_clicked();
	void on_button_prev_clicked();
	bool on_slider_clicked(GdkEventButton* event);
	bool on_slider_released(GdkEventButton* event);
	bool on_slider_value_changed(Gtk::ScrollType type, double value);

};

#endif /* GUISEEKBAR_H_ */
