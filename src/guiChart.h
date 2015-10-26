#ifndef GUICHART_H_
#define GUICHART_H_

#include <iostream>
#include <vector>
#include <gtkmm.h>

#include "guiGraph.h"

/**
 * @class guiChart
 * @brief The chart containing the the zoom, crop and export buttons as well as the graph.
 *        Manages the dataset before it is displayed in the graph.
 */
class guiChart : public Gtk::Frame  {

public:
	guiChart(Controller *c, guiMain *gui);
	virtual ~guiChart();

	void initDataset();
	void updateDataset();
	void setMarkerPosition(int frameID);

	inline bool getActiveSelection() {return graph->getActiveSelection(); }
	uint getSelectionFrom() {return graph->getSelectionFrom(); };
	uint getSelectionTo() {return graph->getSelectionTo(); };

private:

	Controller *controller;
	FrameManager *frameManager;
	FrameProcessor *frameProcessor;
	guiMain* mainGUI;

	guiGraph *graph;
	Timeseries dataset;

	int zoomLevel;
	uint nSamples;

	Gtk::VBox m_VBox_Chart;
	Gtk::HBox m_HBox_ChartControl;
	Gtk::Button m_Button_ZoomIn;
	Gtk::Button m_Button_ZoomOut;
	Gtk::Button m_Button_Crop;
	Gtk::Button m_Button_Export;
	Gtk::CheckButton m_CheckButton_Selection;

	Gtk::Adjustment m_Adjustment_Chart;
	Gtk::HScrollbar m_HScrollbar_Chart;

	std::string m_currentPath;

protected:

	void on_button_zoom_in_clicked();
	void on_button_zoom_out_clicked();
	bool on_slider_value_changed(Gtk::ScrollType type, double value);

	void on_button_crop_clicked();
	void on_button_export_clicked();
	void on_checkbutton_selection_clicked();
};


#endif /* GUICHART_H_ */
