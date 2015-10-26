#include <iostream>
#include <algorithm>

#include "controller.h"
#include "guiGraph.h"
#include "guiMain.h"

guiGraph::guiGraph(Controller *c, guiMain *gui, Timeseries &collection) : collection(collection) {

	controller = c;
	frameManager = controller->getFrameManager();
	frameProcessor = frameManager->getFrameProcessor();
	mainGUI = gui;

	add_events(Gdk::EXPOSURE_MASK);
	add_events(Gdk::BUTTON1_MOTION_MASK);
	add_events(Gdk::BUTTON2_MOTION_MASK);
	add_events(Gdk::BUTTON3_MOTION_MASK);
	add_events(Gdk::BUTTON_PRESS_MASK);
	add_events(Gdk::BUTTON_RELEASE_MASK);

	mouseDragging = false;
	activeSelection = false;
	markerPositionSample = 0;
	markerPositionPixel = 0;

	zoomLevel = 0;
	zoomFactor = std::pow(0.5, zoomLevel); // zoomFactor of 1.0 displays full dataset
	leftBoundary = 0;
	stepSize = 0;
}

void guiGraph::updateSamples(const Timeseries& inSample) {
	collection = inSample;
	nSamples = collection.size() > 0 ? collection.begin()->second.rawData.size() : 0;
	sampleRange = zoomFactor * nSamples; // sample span (left and right boundary of view)
	invalidate();
}


void guiGraph::setZoom(int zoom) {
	zoomLevel = zoom;
	zoomFactor = std::pow(0.5, zoomLevel); // zoomFactor of 1.0 displays full dataset

	int rangeCenter = leftBoundary + sampleRange/2; // Midpoint of current view
	sampleRange = zoomFactor * nSamples; // Range between left and right boundary of next zoom level
	int nextLeftBoundary = rangeCenter - sampleRange/2;

	if(activeSelection) {
		double selectionCenter = selectionFromPixel + selectionWidthPixel/2.0;
		nextLeftBoundary = pixelToSample(selectionCenter)-sampleRange/2;
	} else {
		nextLeftBoundary = pixelToSample(markerPositionPixel)-sampleRange/2;  // Zoom in on marker
	}
	if(sampleRange != 0) {
		stepSize = static_cast<double>(sampleRange-1) / nSubSamples;
	} else {
		stepSize = 0;
	}
	// Invalidate subsampled data
	for(Timeseries::iterator it = collection.begin(); it != collection.end(); ++it) {
		it->second.calculateOverview = true;
		it->second.calculateFiltering = true;
	}
	moveLeftBoundary(nextLeftBoundary);
}


/**
 * Clamp to possible range
 */
int guiGraph::clampSelection(int value) {
	if(value < 0)
		return 0;
	else if(value > nSamples-1)
		return nSamples-1;
	else return value;
}


/**
 * Map selected pixels to actual samples
 */
void guiGraph::updateSelection() {

	// Pixel -> Sample
	selectionStartSample = pixelToSample(selectionStartPixel);
	selectionEndSample = pixelToSample(selectionEndPixel);
	selectionFromPixel = std::min(selectionStartPixel, selectionEndPixel);
	selectionToPixel = std::max(selectionStartPixel, selectionEndPixel);
	selectionFromSample = pixelToSample(selectionFromPixel);
	selectionFromSample = clampSelection(selectionFromSample);
	selectionToSample = pixelToSample(selectionToPixel);
	selectionToSample = clampSelection(selectionToSample);

	// Readjust position in pixel space
	selectionFromPixel = sampleToPixel(selectionFromSample);
	selectionToPixel = sampleToPixel(selectionToSample);
}


/**
 * Map marked pixel to actual sample
 */
void guiGraph::updateMarkerPosition() {
	markerPositionSample = pixelToSample(markerPositionPixel);
	markerPositionPixel = sampleToPixel(markerPositionSample);
	mainGUI->setCurrentFrame(markerPositionSample);
}


/**
 * Map marker from sample- to pixel space
 */
void guiGraph::setMarkerPosition(int frameID) {
	markerPositionSample = frameID;
	markerPositionPixel = sampleToPixel(markerPositionSample);
	invalidate();
}


void guiGraph::setActiveSelection(bool active) {
	activeSelection = active;
}


void guiGraph::moveLeftBoundary(int pos) {

	leftBoundary = std::max(0, pos); // Clamp to lower limit
	leftBoundary = std::min(leftBoundary, nSamples-sampleRange); // Clamp to upper limit

	// Reposition marker/selection
	selectionStartPixel = sampleToPixel(selectionStartSample);
	selectionEndPixel = sampleToPixel(selectionEndSample);
	markerPositionPixel = sampleToPixel(markerPositionSample);
	selectionFromPixel = sampleToPixel(selectionFromSample);
	selectionToPixel = sampleToPixel(selectionToSample);

	// Invalidate subsampled data
	for(Timeseries::iterator it = collection.begin(); it != collection.end(); ++it) {
		it->second.calculateFiltering = true;
	}
	invalidate();
}


int guiGraph::pixelToSample(double pixel) {
	int sample = leftBoundary + static_cast<int>(pixel*stepSize + 0.5);
	return sample;
}


double guiGraph::sampleToPixel(int sample) {
	double pixel = static_cast<double>(sample-leftBoundary)/stepSize;
	return pixel;
}


int guiGraph::getLeftBoundary() {
	return leftBoundary;
}


double guiGraph::getStepSize() {
	return stepSize;
}


int guiGraph::getSampleRange() {
	return sampleRange;
}


void guiGraph::calcSubsamplesOverview(TimeSeriesDataset &timeSeries) {
	printf("calcSubsamplesOverview\n");
	// Multiple samples per pixel
	// Calculate min/max of sample intervals
	timeSeries.sampleIntervalMin.resize(nSubSamples);
	timeSeries.sampleIntervalMax.resize(nSubSamples);
	for(uint i = 0; i < nSubSamples ; i++) {
		int firstElement = leftBoundary + i*stepSize; // Beginning of interval
		int lastElement = firstElement + stepSize; // End of interval
		if(i == nSubSamples-1) {
			lastElement = nSamples-1; // Adjust size of last interval
		}
		std::vector<float>::iterator left = timeSeries.rawData.begin() + firstElement;
		std::vector<float>::iterator right = timeSeries.rawData.begin() + lastElement;
		timeSeries.sampleIntervalMin[i] = *std::min_element(left, right);
		timeSeries.sampleIntervalMax[i] = *std::max_element(left, right);
	}
	timeSeries.calculateOverview = false;

}


void guiGraph::calcSubsamplesFiltering(TimeSeriesDataset &timeSeries) {

	timeSeries.filteredSamples.resize(sampleRange);
	std::vector<float>::iterator from = timeSeries.rawData.begin() + leftBoundary;
	std::vector<float>::iterator to   = timeSeries.rawData.begin() + leftBoundary + sampleRange;
	std::copy(from, to, timeSeries.filteredSamples.begin());

	// Pyramidal linear filtering
	// Repeat last data point in case of odd sample size
	if(timeSeries.filteredSamples.size()%2 != 0) {
		timeSeries.filteredSamples.push_back(timeSeries.filteredSamples.back());
	}
	// Halve sample size with each iteration (linear interpolation)
	double kernel[] = {0.25, 0.5, 0.25};
	int iteration = 0;
	while(timeSeries.filteredSamples.size() > 2*nSubSamples) {
		// Apply 1D-kernel
		double sum;
		for(unsigned int j = 0; j < timeSeries.filteredSamples.size() ; j++) {
			sum = 0.0;
			for(int k = -1; k <= 1; k++) {
				int index;
				// Repeat border samples
				if((j-k) < 0) {
					index = 0; // left border
				} else if((j-k) >= timeSeries.filteredSamples.size()) {
					index = timeSeries.filteredSamples.size()-1; // right border
				} else {
					index = (j-k);
				}
				sum += kernel[k+1] * timeSeries.filteredSamples[index];
			}
			timeSeries.filteredSamples[j] = sum;
		}

		// In-place subsampling
		unsigned int halfSize = timeSeries.filteredSamples.size() / 2;
		for(unsigned int i = 0; i < halfSize; i++) {
			timeSeries.filteredSamples[i] = (timeSeries.filteredSamples[2*i] + timeSeries.filteredSamples[2*i+1])/2.0;
		}
		timeSeries.filteredSamples.resize(halfSize); // Remove now obsolete upper half
		iteration++;
	}

	// Determine scale for final sampling
	double scale;
	if(timeSeries.filteredSamples.size() > nSubSamples) {
		scale = static_cast<double>(timeSeries.filteredSamples.size()) / nSubSamples; // down-sampling
	} else {
		scale = static_cast<double>(nSubSamples) / timeSeries.filteredSamples.size(); // up-sampling
	}

	// Final subsampling
	for(unsigned int i = 0; i < nSubSamples; i++) {
		double indexInterp = scale * i;
		int indexLow = static_cast<int>(indexInterp);
		int indexHigh = indexLow+1;
		double diff = indexInterp - indexLow;
		// Linear interplation: f0 + ((f1 - f0) * x);
		timeSeries.filteredSamples[i] = timeSeries.filteredSamples[indexLow] + (timeSeries.filteredSamples[indexHigh]-timeSeries.filteredSamples[indexLow])*diff;
	}
	timeSeries.filteredSamples.resize(nSubSamples); // Remove upper half
	timeSeries.calculateFiltering = false;
}


bool guiGraph::on_expose_event(GdkEventExpose* event) {

	// This is where we draw on the window
	Glib::RefPtr<Gdk::Window> window = get_window();

	if(window) { // Only run if Window does exist

		// Cairo context
		Cairo::RefPtr<Cairo::Context> cr = window->create_cairo_context();

		// Clip to the area indicated by the expose event so that only a portion of the window that needs to be redrawn
		// Subsamples are still precomputed though
		cr->rectangle(event->area.x, event->area.y,event->area.width, event->area.height);
		cr->clip();

		int widthNew = get_allocation().get_width();
		int heightNew = get_allocation().get_height();

		// Check if widget was resized
		bool resized = false;
		if(widthNew != width) {
			width = widthNew;
			resized = true;
		}
		if(heightNew != height) {
			height = heightNew;
			resized = true;
		}

		nSubSamples = width; // Drawn samples
		double stepSizeNew = static_cast<double>(sampleRange-1) / nSubSamples; // Samples per pixel

		if(stepSizeNew != stepSize) {
			// Recalculate subsamples if step size has changed
			for(Timeseries::iterator it = collection.begin(); it != collection.end(); ++it) {
				it->second.calculateOverview = true;
				it->second.calculateFiltering = true;
			}
		}
		stepSize = stepSizeNew;

		// Reposition marker/selection
		if(resized) {
			markerPositionPixel = sampleToPixel(markerPositionSample);
			selectionFromPixel = sampleToPixel(selectionFromSample);
			selectionToPixel = sampleToPixel(selectionToSample);
		}

		// Draw outline shape
		cr->move_to(0, 0);
		cr->line_to(width, 0);
		cr->line_to(width, height);
		cr->line_to(0, height);
		cr->close_path();
		cr->set_source_rgb(0.1,0.1,0.1);
		cr->fill();

		if(nSamples > 0) {
			float maxValue = 3700; //4095; //*std::max_element(sampleBuffer.begin(), sampleBuffer.end());
			float scaleY = height/maxValue;

			for(Timeseries::iterator it = collection.begin(); it != collection.end(); ++it) {
				TimeSeriesDataset &timeSeries = it->second;

				if(stepSize > 20) {
					// o---------------o
					// | Overview Mode |
					// o---------------o
					//printf("Overview Mode\n");

					if(timeSeries.calculateOverview) {
						calcSubsamplesOverview(timeSeries);
					}

					// Create a path
					double lastCoordYMin = height - timeSeries.sampleIntervalMin[0]*scaleY;
					double lastCoordYMax = height - timeSeries.sampleIntervalMax[0]*scaleY;
					for(uint i = 1; i < nSubSamples; i++) {
						double coordX = width * ( static_cast<double>(i) / nSubSamples); // Note: cairo renders non integer coordinates quite well
						double coordYMin = height - timeSeries.sampleIntervalMin[i]*scaleY; // Note: max is smaller than min due to cairo's coordinate system,
						double coordYMax = height - timeSeries.sampleIntervalMax[i]*scaleY; // but still refers to the larger sample value

						// Connect vertical lines in case of a gap
						if(coordYMin >= lastCoordYMax) {
							coordYMin = lastCoordYMax; // - 1.0;
						}
						if(coordYMax <= lastCoordYMin) {
							coordYMax = lastCoordYMin; // + 1.0;
						}

						// Ensure there is a line of at least one pixel
						if(coordYMin != coordYMax) {
							coordYMin -= 0.5;
							coordYMax += 0.5;
						}
						cr->move_to(coordX, coordYMin);
						cr->line_to(coordX, coordYMax);

						lastCoordYMin = coordYMin;
						lastCoordYMax = coordYMax;
					}
					cr->set_line_width(1.0);
					cr->set_source_rgb(timeSeries.color.r, timeSeries.color.g, timeSeries.color.b);
					cr->stroke();

				} else if(stepSize > 2) {
					// o----------------o
					// | Filtering Mode |
					// o----------------o
					//printf("Filtering Mode\n");
					if(timeSeries.calculateFiltering) {
						calcSubsamplesFiltering(timeSeries);
					}

					// Create a path
					cr->move_to(0, height - timeSeries.rawData[leftBoundary]*scaleY);
					for(unsigned int i = 1; i < timeSeries.filteredSamples.size(); i++) {
						int coordX = width * ( i/static_cast<double>(timeSeries.filteredSamples.size()) );
						int coordY = height - timeSeries.filteredSamples[i]*scaleY;
						cr->line_to(coordX, coordY);
					}
					cr->set_line_width(1.0);
					cr->set_source_rgb(timeSeries.color.r, timeSeries.color.g, timeSeries.color.b);
					cr->stroke();

				} else {
					// o-----------o
					// | Zoom Mode |
					// o-----------o
					//printf("Zoom Mode\n");
					// Create a path
					cr->set_line_width(1.0);
					cr->set_source_rgb(timeSeries.color.r, timeSeries.color.g, timeSeries.color.b);
					cr->move_to(0, height - timeSeries.rawData[leftBoundary]*scaleY);
					for(int i = 1; i < sampleRange; i++) {
						double coordX = width * ( static_cast<double>(i) / (sampleRange-1)); // Note: cairo renders non integer coordinates quite well
						double coordY = height - timeSeries.rawData[leftBoundary+i]*scaleY;
						cr->line_to(coordX, coordY);
					}
					cr->stroke();

					// Data points
					if(stepSize < 0.1) {
						cr->set_source_rgb(timeSeries.color.r, timeSeries.color.g, timeSeries.color.b);
						for(int i = 0; i < sampleRange; i++) {
							double coordX = width * ( static_cast<double>(i) / (sampleRange-1) ); // Note: cairo renders non integer coordinates quite well
							double coordY = height - timeSeries.rawData[leftBoundary+i]*scaleY;
							cr->arc(coordX, coordY, 2.5, 0, 2*M_PI); // Datapoint
							cr->fill();
						}
					}

				}

			}

			// Draw Selection
			if(activeSelection) {
				selectionWidthPixel = selectionToPixel - selectionFromPixel;
				cr->rectangle(selectionFromPixel, 0, selectionWidthPixel, height);
				cr->set_source_rgba(0.0, 1.0, 0.0, 1.0);
				cr->stroke_preserve(); // Border
				cr->set_source_rgba(0.0, 1.0, 0.0, 0.25);
				cr->fill(); // Transparent interior
			}

			// Draw Marker
			cr->move_to(markerPositionPixel, 0); // Top
			cr->line_to(markerPositionPixel, height); // Bottom
			cr->set_line_width(1.0);
			cr->set_source_rgb(1.0, 0.0, 0.0);
			cr->stroke();

		}

	}
	return true;
}


/**
 * Mouse button pressed
 */
bool guiGraph::on_button_press_event(GdkEventButton* event) {

	// Left mouse button
	if(event->button == 1) {
		mouseLeftButtonDown = true;
		if(activeSelection) {
			double range = 0.05 * width;  // Percentage of widget width in pixel space
			double distSelectionStart = abs(event->x - selectionStartPixel);
			double distSelectionEnd = abs(event->x - selectionEndPixel);

			// Redefine selection
			if(distSelectionStart <= distSelectionEnd && distSelectionStart < range) { // Move upper bound
				selectionStartPixel = selectionEndPixel;
				selectionEndPixel = event->x;
			} else if (distSelectionStart > distSelectionEnd && distSelectionEnd < range) { // Move lower bound
				selectionEndPixel = event->x;
			} else {
				activeSelection = false;
				markerPositionPixel = event->x; // Set new marker
				selectionStartPixel = event->x; // Start new selection
			}
		} else {
			markerPositionPixel = event->x; // Set new marker
			selectionStartPixel = event->x;
		}

		updateSelection();
		updateMarkerPosition();
	}

	// Right mouse button
	if(event->button == 3) {}

	return true;
}


/**
 * Mouse button released
 */
bool guiGraph::on_button_release_event(GdkEventButton* event) {
	if(!mouseDragging) {
		activeSelection = false;
		// invalidate();
	}

	mouseDragging = false;

	// Left mouse button
	if(event->button == 1) {
		mouseLeftButtonDown = false;
	}

	// Right mousemouse button
	if(event->button == 3) {}

	return true;
}


/**
 * Moving the mouse with pressed buttons
 */
bool guiGraph::on_motion_notify_event(GdkEventMotion* event) {
	mouseDragging = true;
	if(mouseLeftButtonDown) {
		activeSelection = true;
		selectionEndPixel = event->x;

		// Only redraw if selection actually changed in sample space
		int s1 = selectionFromSample;
		int s2 = selectionToSample;
		updateSelection();
		if( s1 != selectionFromSample || s2 != selectionToSample ) {
			invalidate();
		}
	}
	return true;
}


void guiGraph::forceRedraw() {
	invalidate();
	update();
}


// Invalidate whole window
// Force widget to be redrawn in the near future
void guiGraph::invalidate() {
	Glib::RefPtr<Gdk::Window> window = get_window();
	if(window) {
		Gdk::Rectangle r(0, 0, get_allocation().get_width(), get_allocation().get_height());
		window->invalidate_rect(r, false);
	}
}


// Update window synchronously (fast)
// Causes the redraw to be done immediately
void guiGraph::update() {
	Glib::RefPtr<Gdk::Window> window = get_window();
	if(window) {
		window->process_updates(false);
	}
}


int guiGraph::getSelectionFrom() {
	return selectionFromSample;
}


int guiGraph::getSelectionTo() {
	return selectionToSample;
}
