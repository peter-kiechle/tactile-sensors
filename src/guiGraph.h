#ifndef GUIGRAPH_H_
#define GUIGRAPH_H_

#include <vector>
#include <gtkmm.h>

#include "colormap.h"

struct TimeSeriesDataset {
	std::string description;
	RGB color;
	std::vector<float> rawData;
	std::vector<float> sampleIntervalMin; // Subsampled data: Overview Mode
	std::vector<float> sampleIntervalMax;
	std::vector<float> filteredSamples; // Subsampled data: Filtering mode
	bool calculateOverview;
	bool calculateFiltering;

	TimeSeriesDataset(std::string name, RGB rgb, uint size)
	: description(name),
	  color(rgb),
	  rawData(size),
	  calculateOverview(true),
	  calculateFiltering(true) {}
};

typedef std::map<std::string, TimeSeriesDataset> Timeseries;

// Forward declarations
class Controller;
class FrameManager;
class FrameProcessor;
class guiMain;


/**
 * @class guiGraph
 * @brief The graph.
 * @details The drawing is rather complex and adopts to the number of samples per pixel.
 *          In the overview mode, lines between samples are drawn individually.
 *          Otherwise a pyramidal linear subsampling scheme is applied to draw the time series.
 */
class guiGraph : public Gtk::DrawingArea {

private:

	Controller *controller;
	FrameManager *frameManager;
	FrameProcessor *frameProcessor;
	guiMain *mainGUI;

	Timeseries &collection;

	int width;
	int height;

	bool mouseLeftButtonDown;
	bool mouseDragging;
	bool activeSelection;

	double markerPositionPixel;
	double selectionStartPixel;
	double selectionEndPixel;
	double selectionFromPixel;
	double selectionToPixel;
	double selectionWidthPixel;

	int markerPositionSample;

	// Depend on direction of selection (left to right vs. right to left)
	int selectionStartSample;
	int selectionEndSample;

	// Independent of direction
	int selectionFromSample;
	int selectionToSample;

	int zoomLevel;
	double zoomFactor; // zoomFactor of 1.0 displays full dataset
	int nSamples;
	uint nSubSamples;

	int sampleRange; // sample span (left and right boundary of view)
	double stepSize; // samples per pixel
	int leftBoundary;

	void invalidate();
	void update();
	void forceRedraw();
	int pixelToSample(double pixel);
	double sampleToPixel(int sample);

	int clampSelection(int value);
	void updateSelection();
	void updateMarkerPosition();

	void calcSubsamplesOverview(TimeSeriesDataset &timeSeries);
	void calcSubsamplesFiltering(TimeSeriesDataset &timeSeries);


public:

	guiGraph(Controller *c, guiMain *gui, Timeseries &collection);

	void updateSamples(const Timeseries& inSample);
	void setZoom(int zoom);
	void setMarkerPosition(int frameID);
	void setActiveSelection(bool active);
	inline bool getActiveSelection() {return activeSelection; }

	void moveLeftBoundary(int pos);
	int getLeftBoundary();
	double getStepSize();
	int getSampleRange();
	int getSelectionFrom();
	int getSelectionTo();

protected:

	// Called when our window needs to be redrawn
	virtual bool on_expose_event(GdkEventExpose* event);
	// Called when a mouse button is pressed
	virtual bool on_button_press_event(GdkEventButton* event);
	// Called when a mouse button is released
	virtual bool on_button_release_event(GdkEventButton* event);
	// Called when the mouse moves
	virtual bool on_motion_notify_event(GdkEventMotion* event);

};

#endif /* GUIGRAPH_H_ */
