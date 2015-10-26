#include <cmath>

#include "controller.h"
#include "guiChart.h"
#include "guiMain.h"

guiChart::guiChart(Controller *c, guiMain *gui)
: m_Adjustment_Chart(0, 0, 1, 1, 1, 1)
{
	controller = c;
	frameManager = controller->getFrameManager();
	frameProcessor = frameManager->getFrameProcessor();
	mainGUI = gui;

	graph = Gtk::manage(new guiGraph(controller, mainGUI, dataset));

	zoomLevel = 0;
	nSamples = 0;

	set_shadow_type(Gtk::SHADOW_OUT);

	m_Button_ZoomIn.set_label(Gtk::Stock::ZOOM_IN.id);
	m_Button_ZoomIn.set_use_stock(true);
	m_Button_ZoomIn.signal_clicked().connect( sigc::mem_fun(*this, &guiChart::on_button_zoom_in_clicked) );

	m_Button_ZoomOut.set_label(Gtk::Stock::ZOOM_OUT.id);
	m_Button_ZoomOut.set_use_stock(true);
	m_Button_ZoomOut.signal_clicked().connect( sigc::mem_fun(*this, &guiChart::on_button_zoom_out_clicked) );

	m_Button_Crop.set_label("Crop to selection");
	m_Button_Crop.signal_clicked().connect( sigc::mem_fun(*this, &guiChart::on_button_crop_clicked) );

	m_Button_Export.set_label("Export selection");
	m_Button_Export.signal_clicked().connect( sigc::mem_fun(*this, &guiChart::on_button_export_clicked) );

	m_CheckButton_Selection.set_label("Show selected cells");
	m_CheckButton_Selection.set_active(true);
	m_CheckButton_Selection.signal_clicked().connect( sigc::mem_fun(*this, &guiChart::on_checkbutton_selection_clicked) );

	m_HBox_ChartControl.pack_start(m_Button_ZoomIn, Gtk::PACK_SHRINK, 5);
	m_HBox_ChartControl.pack_start(m_Button_ZoomOut, Gtk::PACK_SHRINK, 5);
	m_HBox_ChartControl.pack_start(m_Button_Crop, Gtk::PACK_SHRINK, 5);
	m_HBox_ChartControl.pack_start(m_Button_Export, Gtk::PACK_SHRINK, 5);
	m_HBox_ChartControl.pack_start(m_CheckButton_Selection, Gtk::PACK_SHRINK, 5);

	m_HScrollbar_Chart.signal_change_value().connect(sigc::mem_fun(*this, &guiChart::on_slider_value_changed));
	m_HScrollbar_Chart.hide();

	m_VBox_Chart.pack_start(m_HBox_ChartControl, Gtk::PACK_SHRINK, 5);
	m_VBox_Chart.pack_start(*graph, Gtk::PACK_EXPAND_WIDGET, 5);
	m_VBox_Chart.pack_start(m_HScrollbar_Chart, Gtk::PACK_SHRINK, 5);

	m_currentPath = controller->getProfileDirectory();

	// add the widget to the window
	add(m_VBox_Chart);
}

guiChart::~guiChart() {};


/**
 * Init chart
 */
void guiChart::initDataset() {
	dataset.clear();
	zoomLevel = 0;
	graph->setZoom(zoomLevel);

	m_Adjustment_Chart.set_lower(0);
	m_Adjustment_Chart.set_upper(nSamples-1);
	m_Adjustment_Chart.set_step_increment(graph->getStepSize());
	m_Adjustment_Chart.set_page_increment(graph->getSampleRange()/2);
	m_Adjustment_Chart.set_page_size(graph->getSampleRange());
	m_Adjustment_Chart.set_value(graph->getLeftBoundary());
	m_HScrollbar_Chart.set_adjustment(m_Adjustment_Chart);
	m_HScrollbar_Chart.show();
}


/**
 * Only collect/copy missing data series
 */
void guiChart::updateDataset() {

	nSamples = frameManager->getFrameCountTS();

	std::vector<std::string> descriptions;

	// Collect selected cells
	if(m_CheckButton_Selection.get_active()) {
		// Collect IDs of selected cells
		std::vector<int> cellIDs = frameManager->createSelectedCellsIdx();
		// Add missing time series of selected cells
		for(uint i = 0; i < cellIDs.size(); i++) {
			uint cellID = cellIDs[i];
			std::ostringstream ss;
			ss << "Cell " << cellID << ":";
			std::string description = ss.str();
			descriptions.push_back(description);
			if(dataset.find(description) == dataset.end() ) { // not in map -> add it
				RGB green(0.2, 1.0, 0.2);
				dataset.insert(std::make_pair(description, TimeSeriesDataset(description, green, nSamples) ));
				TimeSeriesDataset &data = dataset.find(description)->second;
				for(uint frameID = 0; frameID < frameManager->getFrameCountTS(); frameID++) {
					TSFrame* tsFrame = frameManager->getFilteredFrame(frameID);
					data.rawData[frameID] = tsFrame->cells[cellID];;
				}
			}
		}
	}

	// Collect characteristics (min, max, average)
	std::vector<std::vector<int> > characteristics = mainGUI->getCharacteristics();

	// Add missing time series
	for(uint i = 0; i < characteristics.size(); i++) {
		uint m = characteristics[i][0];
		int id = characteristics[i][1];
		std::ostringstream ss;
		RGB color;
		RGB white(1.0, 1.0, 1.0);
		RGB red(1.0, 0.2, 0.2);
		RGB yellow(1.0, 1.0, 0.2);
		if(m < frameManager->getNumMatrices()) { // Matrix Characteristics
			switch(id) {
			case 0:
				ss << "Matrix " << m << ": Average";
				color = white;
				break;
			case 1:
				ss << "Matrix " << m << ": Min";
				color = yellow;
				break;
			case 2:
				ss << "Matrix " << m << ": Max";
				color = red;
				break;
			}
		} else { // Frame Characteristics
			switch(id)	{
			case 0:
				ss << "Average:";
				color = white;
				break;
			case 1:
				ss << "Min";
				color = yellow;
				break;
			case 2:
				ss << "Max";
				color = red;
				break;
			}
		}
		std::string description = ss.str();
		descriptions.push_back(description);
		if(dataset.find(description) == dataset.end() ) { // not in map -> add it
			dataset.insert(std::make_pair(description, TimeSeriesDataset(description, color, nSamples) ));
			TimeSeriesDataset &data = dataset.find(description)->second;
			if(m < frameManager->getNumMatrices()) { // Matrix Characteristics
				switch(id)	{
				case 0:
					for(uint frameID = 0; frameID < frameManager->getFrameCountTS(); frameID++) {
						data.rawData[frameID] = frameProcessor->getMatrixAverage(frameID, m);
					}
					break;
				case 1:
					for(uint frameID = 0; frameID < frameManager->getFrameCountTS(); frameID++) {
						data.rawData[frameID] = frameProcessor->getMatrixMin(frameID, m);
					}
					break;
				case 2:
					for(uint frameID = 0; frameID < frameManager->getFrameCountTS(); frameID++) {
						data.rawData[frameID] = frameProcessor->getMatrixMax(frameID, m);
					}
					break;
				}
			} else {
				switch(id)	{
				case 0:
					for(uint frameID = 0; frameID < frameManager->getFrameCountTS(); frameID++) {
						data.rawData[frameID] = frameProcessor->getAverage(frameID);
					}
					break;
				case 1:
					for(uint frameID = 0; frameID < frameManager->getFrameCountTS(); frameID++) {
						data.rawData[frameID] = frameProcessor->getMin(frameID);
					}
					break;
				case 2:
					for(uint frameID = 0; frameID < frameManager->getFrameCountTS(); frameID++) {
						data.rawData[frameID] = frameProcessor->getMax(frameID);
					}
					break;
				}
			}
		}
	}

	// Remove deselected time series
	for(Timeseries::iterator it = dataset.begin(); it != dataset.end(); /* no increment */) {
		if(std::find(descriptions.begin(), descriptions.end(), it->first) == descriptions.end() ) { // Map element not found -> delete it
			if(dataset.size() > 1) {
				dataset.erase(it++); // Erase before incrementing
			} else {
				dataset.clear();
				break;
			}
		} else {
			++it;
		}
	}

	graph->updateSamples(dataset);

	m_Adjustment_Chart.set_lower(0);
	m_Adjustment_Chart.set_upper(nSamples-1);
	m_Adjustment_Chart.set_step_increment(graph->getStepSize());
	m_Adjustment_Chart.set_page_increment(graph->getSampleRange()/2);
	m_Adjustment_Chart.set_page_size(graph->getSampleRange());
}


void guiChart::setMarkerPosition(int frameID) {
	graph->setMarkerPosition(frameID); // Forward to child
}


void guiChart::on_button_zoom_in_clicked() {
	zoomLevel++;
	graph->setZoom(zoomLevel);
	m_Adjustment_Chart.set_step_increment(graph->getStepSize());
	m_Adjustment_Chart.set_page_increment(graph->getSampleRange()/2);
	m_Adjustment_Chart.set_page_size(graph->getSampleRange());
	m_Adjustment_Chart.set_value(graph->getLeftBoundary());
}


void guiChart::on_button_zoom_out_clicked() {
	if(zoomLevel > 0) {
		zoomLevel--;
		graph->setZoom(zoomLevel);
		m_Adjustment_Chart.set_step_increment(graph->getStepSize());
		m_Adjustment_Chart.set_page_increment(graph->getSampleRange()/2);
		m_Adjustment_Chart.set_page_size(graph->getSampleRange());
		m_Adjustment_Chart.set_value(graph->getLeftBoundary());
	}
}


bool guiChart::on_slider_value_changed(Gtk::ScrollType type, double value) {
	graph->moveLeftBoundary(static_cast<int>( m_Adjustment_Chart.get_value() ));
	return true;
}


void guiChart::on_button_crop_clicked() {
	if(graph->getActiveSelection()) {
		int fromIdx = graph->getSelectionFrom();
		int toIdx = graph->getSelectionTo();

		uint64_t from = frameManager->getFrame(fromIdx)->timestamp;
		uint64_t to = frameManager->getFrame(toIdx)->timestamp;

		printf("Cropping to selection: [%d (%lld) - %d (%lld)]\n", fromIdx, from, toIdx, to);

		frameManager->cropToFrames(from, to);

		graph->setActiveSelection(false);
		graph->setZoom(0);
		dataset.clear();
		//updateDataset();
		mainGUI->updateGUIOffline();
	}
}


// Note: There is a bug in older versions of GTK resulting in:
// Gtk-WARNING **: Unable to retrieve the file info
void guiChart::on_button_export_clicked() {
	if(graph->getActiveSelection()) {
		int fromIdx = graph->getSelectionFrom();
		int toIdx = graph->getSelectionTo();

		std::ostringstream ss;
		std::string basename = controller->getProfileBaseName();

		ss << basename << "_" << std::setfill('0') << std::setw(6) << fromIdx << "-" << std::setw(6) << toIdx << ".dsa";
		std::string filename = ss.str();

		Gtk::FileChooserDialog dialog("Export Pressure Profile", Gtk::FILE_CHOOSER_ACTION_SAVE);
		dialog.set_transient_for(*mainGUI);
		dialog.set_current_folder(m_currentPath);
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
				std::string filename = dialog.get_filename();
				m_currentPath = dialog.get_current_folder();

				//printf("Exporting selection: [%d, %d]\n", fromIdx, toIdx);

				frameManager->storeProfileSelection(filename, fromIdx, toIdx);

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
}


void guiChart::on_checkbutton_selection_clicked() {
	updateDataset();
}
