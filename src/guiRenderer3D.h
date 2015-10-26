#ifndef RENDERER3D_H_
#define RENDERER3D_H_

#include <gtkmm.h>
#include <gtkglmm.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>

#include "guiRenderer.h"
#include "colormap.h"
#include "camera.h"
#include "utils.h"
#include "forwardKinematics.h"

/**
 * @class guiRenderer3D
 * @brief Renders visualization of tactile sensor profiles in OpenGL.
 * @details Uses the old OpenGL immediate mode.
 *          Given the (recorded) joint angles the computed miniball can be visualized.
 */
class guiRenderer3D : public guiRenderer,
public Gtk::DrawingArea,
public Gtk::GL::Widget<guiRenderer3D> // Our own Widget
{

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	explicit guiRenderer3D(FrameManager* frameManager);
	explicit guiRenderer3D(FrameManager* frameManager, guiMain *gui);
	virtual ~guiRenderer3D();

	virtual void renderFrame();
	virtual void renderFrame(uint frameID);
	void takeScreenshot(std::string filename);
	void takeScreenshot(int width, int height, std::string filename);
	void setOffscreenSize(int width, int height);
	void init(); // build sensor layout

	// Called when any key on the keyboard is pressed
	bool on_key_press_event(GdkEventKey* event);
	// Called when any key on the keyboard is released
	bool on_key_release_event(GdkEventKey* event);

private:

	int m_widgetWidth;
	int m_widgetHeight;

	std::vector<double> m_matrixCenterX;
	std::vector<double> m_matrixCenterY;
	std::vector<double> m_topLeftCellCenterX;
	std::vector<double> m_topLeftCellCenterY;
	std::vector<double> m_cellTopLeftX;
	std::vector<double> m_cellTopLeftY;
	std::vector<double> m_cellBottomRightX;
	std::vector<double> m_cellBottomRightY;

	double m_leftMargin;
	double m_rightMargin;
	double m_topMargin;
	double m_bottomMargin;

	double m_sensorWidth;
	double m_sensorHeight;

	double m_scaleFactor;
	double m_offsetX;
	double m_offsetY;


	// Rendering modes
	bool m_drawNormals;
	bool m_wireFrameMode;

	enum renderMode { MATRICES_2D_FLAT, MATRICES_2D_BARS, MATRICES_3D_CELLS, MATRICES_3D_POINTCLOUD };
	renderMode m_mode;

	enum renderModeMiniball { MB_NONE, MB_OPAQUE, MB_TRANSPARENT, MB_WIREFRAME };
	renderModeMiniball m_modeMiniball;

	// Free view camera
	boost::shared_ptr<Camera> m_camera;
	float m_speedFactor; // movement speed
	bool m_isMoving; // active animation
	bool m_isLooping; // active Render loop

	std::vector<bool> m_isVisible;

	double m_currentTime;
	double m_targetTime;
	double m_cameraDelay; // Camera refresh rate in ms

	Eigen::Vector3d m_pos2D; // Camera vectors for 2D render modes
	Eigen::Vector3d m_view2D;
	Eigen::Vector3d m_pos2D_default;
	Eigen::Vector3d m_view2D_default;

	Eigen::Vector3d m_pos3D; // Camera vectors for 3D render modes
	Eigen::Vector3d m_view3D;
	Eigen::Vector3d m_pos3D_default;
	Eigen::Vector3d m_view3D_default;

	bool m_cameraMoveLeft;
	bool m_cameraMoveRight;
	bool m_cameraMoveUp;
	bool m_cameraMoveDown;
	bool m_cameraMoveForward;
	bool m_cameraMoveBackward;

	// Mouse movement
	double m_mouseSensitivity;

	bool m_mouseLeftButtonDown;
	bool m_mouseRightButtonDown;
	bool m_mouseMiddleButtonDown;

	bool m_mouseDragging; // Movement with pressed mouse button
	bool m_selectionMode;
	bool m_previousSelectionState;

	Eigen::Vector2i m_mouseLeftPressed; // Mouse position during last left click
	Eigen::Vector2i m_mouseRightPressed; // Mouse position during last left click
	Eigen::Vector2i m_mouseMiddlePressed; // Mouse position during last left click

	Eigen::Vector2i m_rotationDelta;

	sigc::connection m_cameraMovementConnection;
	sigc::connection m_ConnectionIdle;

	Eigen::Vector4f m_lightPosition0;
	Eigen::Vector4f m_lightPosition1;
	Eigen::Vector4f m_lightPosition2;
	Eigen::Vector4f m_lightPosition3;

	ForwardKinematics m_forwardKinematics;

	// Off-screen rendering
	Glib::RefPtr<Gdk::GL::Config> m_GLConfig_Offscreen;
	Glib::RefPtr<Gdk::Pixmap> m_Pixmap_Offscreen;
	int m_widthOffscreen;
	int m_heightOffscreen;

	void delegateConstructor();

	void moveCamera();
	void rotateCamera();
	void updateCamera();

	bool renderLoop(); // Callback function the timeout will call. Used for fps controlled movement

	void initMatrices2D();
	void drawMatrices();
	void drawMatrices2DFlat();
	void drawMatrices2DBars();
	void drawMatrices3DCells();
	void drawMatrices3DPointCloud();

	void drawLine(double x1, double y1, double z1, double x2, double y2, double z2, double diameter);
	void drawArrow(double x1, double y1, double z1, double x2, double y2, double z2, double diameter);
	void drawCoordinateAxes(double length);
	void drawMinmalBoundingSphere(std::vector<std::vector<double> >& activePoints);
	void drawGrid(float scale);

	void determineSelection(int x, int y, int deltaX, int deltaY);
	void listSelection(GLint hits, GLuint *names);

	void initOffscreen();

	int printOglError(const char *file, int line);

protected:

	virtual void invalidate(); // Has to be overwritten
	virtual void update(); // Has to be overwritten

	//Override default signal handlers:
	// Called when GL is first initialized
	virtual void on_realize();
	// Called when our window needs to be redrawn
	virtual bool on_expose_event(GdkEventExpose* event);
	// Called when the window is resized
	virtual bool on_configure_event(GdkEventConfigure* event);

	// Called when a mouse button is pressed
	virtual bool on_button_press_event(GdkEventButton* event);
	// Called when a mouse button is released
	virtual bool on_button_release_event(GdkEventButton* event);
	// Called when the mouse moves
	virtual bool on_motion_notify_event(GdkEventMotion* event);

};

#endif /* RENDERER3D_H_ */
