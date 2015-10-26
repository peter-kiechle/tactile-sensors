#ifndef GUISCREENSHOT_H
#define GUISCREENSHOT_H

#include <gtkmm.h>
#include <glibmm.h>

#include "controller.h"
#include "framemanager.h"

class guiRenderer3D;
class FrameManager;

/**
 * @class guiScreenshot
 * @brief Take screen shot GUI.
 */
class guiScreenshot : public Gtk::Window {
public:
	guiScreenshot(Controller *c, guiRenderer3D* renderer, uint from, uint to);
	virtual ~guiScreenshot();

private:
	Controller *m_controller;
	FrameManager *m_frameManager;
	guiRenderer3D *m_guiRenderer3D;
	uint m_startFrame;
	uint m_stopFrame;

protected:
	void on_button_render_clicked(); 
	void on_button_close_clicked(); 
	Gtk::Button m_Button_Render;
	Gtk::Button m_Button_Close;

	Gtk::Label m_Label_Width;
	Gtk::Label m_Label_Height;
	Gtk::Label m_Label_From;
	Gtk::Label m_Label_To;
	Gtk::Label m_Label_From_Value;
	Gtk::Label m_Label_To_Value;

	Gtk::Adjustment m_Adjustment_Width;
	Gtk::SpinButton m_SpinButton_Width;
	Gtk::Adjustment m_Adjustment_Height;
	Gtk::SpinButton m_SpinButton_Height;

	Gtk::Table m_Table;

	Gtk::HButtonBox m_ButtonBox_Dialog;
	Gtk::VBox m_VBox_Dialog;

};

#endif //GUISCREENSHOT_H
