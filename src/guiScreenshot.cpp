#include <iostream>
#include <string>
#include <sstream>

#include "guiScreenshot.h"
#include "guiRenderer3D.h"

guiScreenshot::guiScreenshot(Controller *c, guiRenderer3D* renderer, uint from, uint to)
:
m_Adjustment_Width(2048.0, 50.0, 5000.0, 1.0, 0.0, 0.0),
m_SpinButton_Width(m_Adjustment_Width),
m_Adjustment_Height(2048.0, 50.0, 5000.0, 1.0, 0.0, 0.0),
m_SpinButton_Height(m_Adjustment_Height),
m_Table(4, 2, true) {

	m_controller = c;
	m_frameManager = m_controller->getFrameManager();
	m_guiRenderer3D = renderer;
	m_startFrame = from;
	m_stopFrame = to;

	std::ostringstream ss;
	ss << m_startFrame;
	std::string startFrame_str = ss.str();
	ss.str(""); // clear
	ss << m_stopFrame;
	std::string stopFrame_str = ss.str();

	set_title("Screenshot 3D");
	set_border_width(5);
	set_position(Gtk::WIN_POS_CENTER);

	m_Label_Width.set_text("Width:");
	m_Label_Width.set_width_chars(10);
	m_Label_Width.set_alignment(1, 0.5);

	m_Label_Height.set_text("Height:");
	m_Label_Height.set_alignment(1, 0.5);

	m_Label_From.set_text("Start frame:");
	m_Label_From.set_alignment(1, 0.5);
	m_Label_From_Value.set_text(startFrame_str);
	m_Label_From_Value.set_alignment(0, 0.5);

	m_Label_To.set_text("Stop frame:");
	m_Label_To.set_alignment(1, 0.5);
	m_Label_To_Value.set_text(stopFrame_str);
	m_Label_To_Value.set_alignment(0, 0.5);

	// left_attach, right_attach, top_attach, bottom_attach
	m_Table.attach(m_Label_Width, 0, 1, 0, 1, Gtk::FILL, Gtk::FILL, 10, 5);
	m_Table.attach(m_Label_Height, 0, 1, 1, 2, Gtk::FILL, Gtk::FILL, 10, 5);
	m_Table.attach(m_Label_From, 0, 1, 2, 3, Gtk::FILL, Gtk::FILL, 10, 5);
	m_Table.attach(m_Label_To, 0, 1, 3, 4, Gtk::FILL, Gtk::FILL, 10, 5);

	m_Table.attach(m_SpinButton_Width, 1, 2, 0, 1);
	m_Table.attach(m_SpinButton_Height, 1, 2, 1, 2);
	m_Table.attach(m_Label_From_Value, 1, 2, 2, 3);
	m_Table.attach(m_Label_To_Value, 1, 2, 3, 4);

	m_Button_Close.set_label("Close");
	m_Button_Close.signal_clicked().connect(sigc::mem_fun(*this, &guiScreenshot::on_button_close_clicked));

	m_Button_Render.set_label("Render");
	m_Button_Render.signal_clicked().connect(sigc::mem_fun(*this, &guiScreenshot::on_button_render_clicked));

	m_ButtonBox_Dialog.pack_start(m_Button_Close, Gtk::PACK_SHRINK, 5);
	m_ButtonBox_Dialog.pack_start(m_Button_Render, Gtk::PACK_SHRINK, 5);
	m_ButtonBox_Dialog.set_border_width(5);

	m_VBox_Dialog.pack_start(m_Table);
	m_VBox_Dialog.pack_start(m_ButtonBox_Dialog);

	add(m_VBox_Dialog);

	show_all();
}

guiScreenshot::~guiScreenshot() { }

void guiScreenshot::on_button_close_clicked() {
	hide();
}

void guiScreenshot::on_button_render_clicked() {

	int width = m_SpinButton_Width.get_value_as_int();
	int height = m_SpinButton_Height.get_value_as_int();

	// Determine folder / filename
	std::string currentPath = m_controller->getProfileDirectory();
	std::string basename = m_controller->getProfileBaseName();
	std::string filename;

	if(m_startFrame == m_stopFrame) { // Single frame
		std::ostringstream ss;
		ss << basename << "_frame_" << std::setfill('0') << std::setw(6) << m_startFrame << ".png";
		filename = ss.str();

		Gtk::FileChooserDialog dialog("Save frame as PNG", Gtk::FILE_CHOOSER_ACTION_SAVE);
		//dialog.set_transient_for(*mainGUI);
		dialog.set_current_folder(currentPath);
		dialog.set_current_name(filename);
		dialog.set_do_overwrite_confirmation(true); // Ask for confirmation before overwriting existing file

		//Add response buttons the the dialog:
		dialog.add_button(Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL);
		dialog.add_button(Gtk::Stock::SAVE, Gtk::RESPONSE_OK);

		// Show the dialog and wait for a user response:
		int result = dialog.run();

		// Handle the response
		switch(result) {
			case(Gtk::RESPONSE_OK):
			{
				// Notice that this is a std::string, not a Glib::ustring.
				filename = dialog.get_filename();
				currentPath = dialog.get_current_folder();

				break;
			}
			case(Gtk::RESPONSE_CANCEL):
			{
				break;
			}
			default:
			{
				break;
			}
		}

	} else {

		Gtk::FileChooserDialog dialog("Select folder to save sequence", Gtk::FILE_CHOOSER_ACTION_SELECT_FOLDER);
		//dialog.set_transient_for(*mainGUI);
		dialog.set_current_folder(currentPath);

		//Add response buttons the the dialog:
		dialog.add_button(Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL);
		dialog.add_button(Gtk::Stock::SAVE, Gtk::RESPONSE_OK);

		// Show the dialog and wait for a user response:
		int result = dialog.run();

		// Handle the response
		switch(result) {
			case(Gtk::RESPONSE_OK):
			{
				// Notice that this is a std::string, not a Glib::ustring.
				currentPath = dialog.get_current_folder();
				break;
			}
			case(Gtk::RESPONSE_CANCEL):
			{
				break;
			}
			default:
			{
				break;
			}
		}
	}

	// Take screenshots
	if(m_startFrame == m_stopFrame) {
		m_frameManager->setCurrentFrameID(m_startFrame);
		m_guiRenderer3D->takeScreenshot(width, height, filename);

	} else {
		std::ostringstream ss;
		for(uint frameID = m_startFrame; frameID < m_stopFrame; frameID++) {
			ss.str(""); // clear
			ss << currentPath << "/" << basename << "_frame_" << std::setfill('0') << std::setw(5) << frameID << ".png";
			filename = ss.str();
			m_frameManager->setCurrentFrameID(frameID);
			m_guiRenderer3D->takeScreenshot(width, height, filename);
		}
	}

}
