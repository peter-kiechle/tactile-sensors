#include "guiRenderer.h"

using namespace std;

bool guiRenderer::on_map_event(GdkEventAny* event) {
	if(m_liveMode) {
		if(!m_ConnectionIdle.connected()) {
			m_ConnectionIdle = Glib::signal_idle().connect(sigc::mem_fun(*this, &guiRenderer::on_idle), Glib::PRIORITY_LOW);
		}
	}
	return true;
}


bool guiRenderer::on_unmap_event(GdkEventAny* event) {
	if(m_ConnectionIdle.connected()) {
		m_ConnectionIdle.disconnect();
	}
	return true;
}


bool guiRenderer::on_visibility_notify_event(GdkEventVisibility* event) {
	if (event->state == GDK_VISIBILITY_FULLY_OBSCURED) {
		if(m_ConnectionIdle.connected()) {
			m_ConnectionIdle.disconnect();
		}
	} else {
		if (!m_ConnectionIdle.connected()) {
			if(m_liveMode) {
				m_ConnectionIdle = Glib::signal_idle().connect(sigc::mem_fun(*this, &guiRenderer::on_idle), Glib::PRIORITY_LOW);
			}
		}
	}
	return true;
}


bool guiRenderer::on_idle() {

	// Invalidate the whole window.
	invalidate();

	return true;
}


void guiRenderer::startRendering(bool live) {
	m_liveMode = live;
	m_isRendering = true;

	if(m_liveMode) {
		// Start render loop
		if(!m_ConnectionIdle.connected()) {
			m_ConnectionIdle = Glib::signal_idle().connect(sigc::mem_fun(*this, &guiRenderer::on_idle), Glib::PRIORITY_LOW);
		}
	} else {
		invalidate();
	}
}


void guiRenderer::stopRendering() {
	m_isRendering = false;

	// Stop render loop
	if(m_ConnectionIdle.connected()) {
		m_ConnectionIdle.disconnect();
	}
}


RGB guiRenderer::determineColor(float value) {

	if(utils::almostEqual(value, 0.0, 4)) {
		//RGB color(0.0, 0.61568, 0.87843); // Light blue
		//RGB color(0.04314, 0.17255, 0.33333); // Dark blue
		//RGB color(0.0, 0.0, 0.0); // Black
		//RGB color(0.11764705, 0.13333333, 0.1294117); // Dirty grey
		//RGB color(0.3, 0.3, 0.3); // Grey
		RGB color(0.5, 0.5, 0.5); // Bright Grey
		return color;
	} else {
		RGB color = m_colormap.getColorFromTable(static_cast<int>(value+0.5));
		return color;
	}
}

bool guiRenderer::on_key_press_event(GdkEventKey* event) { return true; } // Workaround for linking error
bool guiRenderer::on_key_release_event(GdkEventKey* event) { return true; }
