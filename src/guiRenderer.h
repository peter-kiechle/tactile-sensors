#ifndef GUIRENDERER_H_
#define GUIRENDERER_H_

#include <gtkmm.h>

#include "colormap.h"
#include "utils.h"
#include "framemanager.h"
#include "frameprocessor.h"

// Forward declaration
class guiMain;

/**
 * @class guiRenderer
 * @brief Base class for guiRenderer2D and guiRenderer3D.
 */
class guiRenderer
{

public:
	explicit guiRenderer(FrameManager* fm) { };
	explicit guiRenderer(FrameManager* fm, guiMain *gui) { };
	virtual ~guiRenderer() { };

	void startRendering(bool live);
	void stopRendering();

	virtual void invalidate() { }; // Has to be overwritten
	virtual void update() { }; // Has to be overwritten

	virtual void renderFrame() { }; // Has to be overwritten
	virtual void renderFrame(uint frameID) { }; // Has to be overwritten


	RGB determineColor(float value);

	FrameManager* m_frameManager;
	FrameProcessor *m_frameProcessor;
	guiMain* m_mainGUI;

	Colormap m_colormap;
	bool m_liveMode;
	bool m_isRendering;

	// Keyboard events are processed first in the top-level widget
	// Called when any key on the keyboard is pressed
	virtual bool on_key_press_event(GdkEventKey* event);
	// Called when any key on the keyboard is released
	virtual bool on_key_release_event(GdkEventKey* event);

protected:

	sigc::connection m_ConnectionIdle;

	//Override default signal handlers:
	virtual bool on_idle();

	// A window is mapped when it becomes visible on the screen
	virtual bool on_map_event(GdkEventAny* event);
	// A window is unmapped when it becomes invisible on the screen.
	virtual bool on_unmap_event(GdkEventAny* event);
	// will be emitted when the widget's window is obscured or unobscured.
	virtual bool on_visibility_notify_event(GdkEventVisibility* event);

};

#endif /* GUIRENDERER_H_ */
