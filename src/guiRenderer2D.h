#ifndef GUIRENDERER2D_H_
#define GUIRENDERER2D_H_

#include <gtkmm/drawingarea.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "guiRenderer.h"
#include "colormap.h"

/**
 * @class guiRenderer2D
 * @brief Renders visualization of tactile sensor profiles with the help of Cairo, a vector graphics library.
 * @details Not very suitable for real-time rendering, but great for rendering PDFs.
 */
class guiRenderer2D : public guiRenderer,
public Gtk::DrawingArea
{
public:
	explicit guiRenderer2D(FrameManager* fm); // Constructor without guiMain
	explicit guiRenderer2D(FrameManager* fm, guiMain *gui); // Constructor with guiMain
	virtual ~guiRenderer2D();

	virtual void renderFrame(); // Has to be overwritten
	virtual void renderFrame(uint frameID); // Has to be overwritten

	void drawMatrices(const Cairo::RefPtr<Cairo::Context>& cr, int width, int height, bool screenshot);
	void takeScreenshot(const string& filename);

	void init(); // build sensor layout

private:

	int m_widgetWidth;
	int m_widgetHeight;

	std::vector<double> m_newCenterX;
	std::vector<double> m_newCenterY;

	std::vector<double> m_matrixCellCenterX;
	std::vector<double> m_matrixCellCenterY;
	std::vector<double> m_rectangleTopLeftX;
	std::vector<double> m_rectangleTopLeftY;
	std::vector<double> m_rectangleBottomRightX;
	std::vector<double> m_rectangleBottomRightY;
	std::vector<double> m_rectangleWidth;
	std::vector<double> m_rectangleHeight;

	double m_leftMargin;
	double m_rightMargin;
	double m_topMargin;
	double m_bottomMargin;

	double m_sensorWidth;
	double m_sensorHeight;

	double m_scaleFactor;
	double m_offsetX;
	double m_offsetY;


	Gtk::EventBox m_EventBox;
	Gtk::Menu m_Menu_Popup;

	bool m_mouseLeftButtonDown;
	bool m_mouseRightButtonDown;
	bool m_mouseDragging; // Movement with pressed mouse button
	bool m_selectionMode;
	Eigen::Vector2i m_mouseLeftPressed; // Mouse position during last left click
	Eigen::Vector2i m_mouseRightPressed; // Mouse position during last left click

	bool m_previousSelectionState;
	void determineSelection(int x, int y);

	bool m_layoutAvailable;
	void rescaleSensorLayout(int width, int height);

	// Called when any key on the keyboard is pressed
	bool on_key_press_event(GdkEventKey* event);
	// Called when any key on the keyboard is released
	bool on_key_release_event(GdkEventKey* event);

	// Right click popup menu
	void on_menu_popup_set_mask();
	void on_menu_popup_reset_mask();

protected:

	virtual void invalidate(); // Has to be overwritten
	virtual void update(); // Has to be overwritten

	//Override default signal handler:
	virtual bool on_expose_event(GdkEventExpose* event);

	// Called when a mouse button is pressed
	virtual bool on_button_press_event(GdkEventButton* event);
	// Called when a mouse button is released
	virtual bool on_button_release_event(GdkEventButton* event);
	// Called when the mouse moves
	virtual bool on_motion_notify_event(GdkEventMotion* event);

};

#endif /* GUIRENDERER2D_H_ */
