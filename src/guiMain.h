#ifndef GUIMAIN_H_
#define GUIMAIN_H_

#include <gtkmm.h>

#include <boost/python.hpp>

#include "controller.h"
#include "guiController.h"
#include "featureExtraction.h"

#include "guiRenderer2D.h"
#include "guiRenderer3D.h"
#include "guiSeekbar.h"
#include "guiChart.h"
#include "guiTreeView.h"
#include "guiTools.h"
#include "guiSlipDetection.h"
#include "guiScreenshot.h"


// Forward declarations
class Controller;
class FrameManager;
class FrameProcessor;
class guiController;
class guiSeekbar;
class guiTools;
class guiSlipDetection;

namespace bp = boost::python;

enum Renderer { RENDERER_CAIRO, RENDERER_OPENGL };


/**
 * @class guiMain
 * @brief The main window. Inherits from Gtk::Window.
 */
class guiMain: public Gtk::Window {

public:
	guiMain(Controller *controller);
	virtual ~guiMain();

	void resetGUIOnline();

	void resetGUIOffline();
	void updateGUIOffline();

	void setCurrentFrame(int frameID);
	void updateDataset();

	inline bool getActiveSelection() {
		return ( getSelectionFrom() >= 0 && getSelectionTo() > 0 && m_guiChart->getActiveSelection() );
	}
	inline uint getSelectionFrom() {return m_guiChart->getSelectionFrom(); };
	inline uint getSelectionTo() {return m_guiChart->getSelectionTo(); };

	void setCharacteristics(std::vector<std::vector<int> > c);
	std::vector<std::vector<int> > getCharacteristics();

	void saveCurrentFramePDF();


protected:
	Controller *m_controller;
	FrameManager *m_frameManager;
	FrameProcessor *m_frameProcessor;
	FeatureExtraction m_featureExtractor;

	uint current_frame;

	Gtk::VBox m_VBox_Main;
	Gtk::MenuBar m_Menubar;
	Gtk::Toolbar m_Toolbar;
	Gtk::ToggleToolButton m_ToggleToolButton_Connect_SDH;
	Gtk::ToggleToolButton m_ToggleToolButton_Connect_DSA;
	bool m_ToggleToolButton_Connect_SDH_pressed; // Work-around for signal emitting set_active()
	bool m_ToggleToolButton_Connect_DSA_pressed;

	Gtk::ToggleToolButton m_ToggleToolButton_Tools;
	Gtk::ToggleToolButton m_ToggleToolButton_Slip_Detection;

	Gtk::ToggleToolButton m_ToggleToolButton_Sensor_View;
	Gtk::ToggleToolButton m_ToggleToolButton_Chart_View;
	Gtk::ToggleToolButton m_ToggleToolButton_Tree_View;

	Gtk::HBox m_HBox_Main;
	Gtk::VBox m_VBox_Right_Sidebar;
	Gtk::VBox m_VBox_Renderer;

	guiController *m_Frame_Controller;

	bool showSensorView;
	bool showChartView;
	bool showTreeView;

	bool m_pythonEmbedded;
	bp::object m_main; // python main module
	bp::object m_global; // python main namespace

	Renderer renderer;


	Gtk::VPaned m_VPaned_Views;
	int m_VPaned_Views_Divider_Pos;
	double m_VPaned_Views_Ratio;
	bool m_resized;

	Gtk::Notebook m_Notebook_Renderer;
	Gtk::Frame m_Frame_Renderer2D;
	guiRenderer2D *m_guiRenderer2D;
	Gtk::Frame m_Frame_Renderer3D;
	guiRenderer3D *m_guiRenderer3D;

	guiSeekbar *m_guiSeekbar;

	guiChart *m_guiChart;

	guiTreeView *m_guiTreeView;

	guiTools *m_guiTools;

	guiSlipDetection *m_guiSlipDetection;

	guiScreenshot* m_guiScreenshot;

	std::vector<std::vector<int> > characteristics;

	// Menu/Toolbar actions
	void on_menu_take_screenshot_2D_clicked();
	void on_menu_take_screenshot_3D_clicked();

	void on_screenshot_delete_clicked();

	void on_menu_connect_SDH();
	void on_menu_connect_DSA();
	void on_menu_new_profile();
	void on_menu_load_profile();
	void on_menu_save_profile_as();
	void on_menu_file_quit();
	void on_menu_show_tools();
	void on_menu_show_slip_detection();
	void on_menu_show_sensor_view();
	void on_menu_show_chart_view();
	void on_menu_show_tree_view();

	void embedPython();
	void on_menu_classify();

	void on_notebook_switch_page(GtkNotebookPage* page, guint page_num);
	void on_vpaned_size_allocate(Gtk::Allocation& allocation);
	void on_vpaned_realize();
	void on_resize_notify(GdkEventConfigure* event);

	bool on_tools_delete_clicked(GdkEventAny* event);
	bool on_slip_detection_delete_clicked(GdkEventAny* event);

	// Keyboard events are processed first in the top-level widget
	// Called when any key on the keyboard is pressed
	virtual bool on_key_press_event(GdkEventKey* event);
	// Called when any key on the keyboard is released
	virtual bool on_key_release_event(GdkEventKey* event);

};

#endif /* GUIMAIN_H_ */
