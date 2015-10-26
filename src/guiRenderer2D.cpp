#include <ctime>
#include <cmath>
#include <algorithm>
#include <cairomm/context.h>

#include "guiRenderer2D.h"
#include "guiMain.h"

guiRenderer2D::guiRenderer2D(FrameManager* fm) : guiRenderer(fm){
	m_frameManager = fm;
	m_frameProcessor = m_frameManager->getFrameProcessor();
	m_mainGUI = NULL;
}

guiRenderer2D::guiRenderer2D(FrameManager* fm, guiMain *gui) : guiRenderer(fm, gui){
	m_frameManager = fm;
	m_frameProcessor = m_frameManager->getFrameProcessor();
	m_mainGUI = gui;
}

guiRenderer2D::~guiRenderer2D() { }


void guiRenderer2D::init() {

	// Register the fact that we want to receive these events
	add_events(Gdk::BUTTON1_MOTION_MASK);
	add_events(Gdk::BUTTON2_MOTION_MASK);
	add_events(Gdk::BUTTON3_MOTION_MASK);
	add_events(Gdk::BUTTON_PRESS_MASK);
	add_events(Gdk::BUTTON_RELEASE_MASK);
	add_events(Gdk::VISIBILITY_NOTIFY_MASK);
	add_events(Gdk::KEY_PRESS_MASK);
	add_events(Gdk::KEY_RELEASE_MASK);

	m_liveMode = false;

	if(m_frameManager->getSensorInfo().nb_matrices == 0) {
		m_isRendering = false;
	}

	//m_colormap.createColorTable(YELLOW_RED, 3700);
	m_colormap.createColorTable(BREWER_YlOrRd, 3700);
	//m_colormap.createColorTable(EXPERIMENTAL, 70);

	m_widgetWidth = 0;
	m_widgetHeight = 0;
	m_layoutAvailable = false;

	m_selectionMode = false;

	if(m_mainGUI) {
		//Fill popup menu:
		Gtk::Menu::MenuList& menulist = m_Menu_Popup.items();
		menulist.push_back( Gtk::Menu_Helpers::MenuElem("Export current frame as PDF", sigc::mem_fun(m_mainGUI, &guiMain::saveCurrentFramePDF) ) );
		menulist.push_back( Gtk::Menu_Helpers::MenuElem("Set dynamic mask to selection", sigc::mem_fun(*this, &guiRenderer2D::on_menu_popup_set_mask) ) );
		menulist.push_back( Gtk::Menu_Helpers::MenuElem("Reset dynamic mask", sigc::mem_fun(*this, &guiRenderer2D::on_menu_popup_reset_mask) ) );
		m_Menu_Popup.accelerate(*this);
	}

	// Build coordinates of sensor cells
	std::vector<double> centerX;
	std::vector<double> centerY;

	std::vector<double> dimensionX;
	std::vector<double> dimensionY;

	// Collect sensor layout data
	for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {
		centerX.push_back(m_frameManager->getMatrixInfo(m).matrix_center_x);
		centerY.push_back(m_frameManager->getMatrixInfo(m).matrix_center_y);
		dimensionX.push_back(m_frameManager->getMatrixInfo(m).cells_x * m_frameManager->getMatrixInfo(m).texel_width);
		dimensionY.push_back(m_frameManager->getMatrixInfo(m).cells_y * m_frameManager->getMatrixInfo(m).texel_height);
	}

	double minX = *std::min_element(centerX.begin(), centerX.end());
	double maxX = *std::max_element(centerX.begin(), centerX.end());
	double minY = *std::min_element(centerY.begin(), centerY.end());
	double maxY = *std::max_element(centerY.begin(), centerY.end());
	double rangeX = maxX-minX;
	double rangeY = maxY-minY;

	// Map cell centers to position in grid
	int nBinsX = 3;
	int nBinsY = 2;
	double shiftX = (minX < 0) ? minX : -minX;
	double shiftY = (minY < 0) ? minY : -minY;

	std::vector<int> gridIndexX;
	std::vector<int> gridIndexY;

	for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {
		int gridX = floor((nBinsX/rangeX)*(centerX[m]-shiftX)); // Index into Grid
		int gridY = ceil(-(nBinsY/rangeY)*(centerY[m]-shiftY)+nBinsY-1); // Reverse order of y-axis

		// clamp values to range [0, nBins-1]
		gridX = gridX > nBinsX-1 ? nBinsX-1 : gridX;
		gridX = gridX < 0 ? 0 : gridX;
		gridY = gridY > nBinsY-1 ? nBinsY-1 : gridY;
		gridY = gridY < 0 ? 0 : gridY;
		gridIndexX.push_back(gridX);
		gridIndexY.push_back(gridY);
	}

	// Calculate new sensor matrix centers
	m_newCenterX.resize(m_frameManager->getNumMatrices());
	m_newCenterY.resize(m_frameManager->getNumMatrices());

	double sensorSpacerX = m_frameManager->getMatrixInfo(0).texel_width/2.0; // Space between sensor matrices
	double sensorSpacerY = m_frameManager->getMatrixInfo(0).texel_height/2.0;

	double gridCenterX[nBinsX];
	double gridCenterY[nBinsY];
	double runningDimX = 0;
	double runningDimY = 0;

	for(int x = 0; x < nBinsX; x++) {
		std::vector<double> gridDimensions;
		for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {
			if(gridIndexX[m] == x) {
				gridDimensions.push_back(dimensionX[m]);
			}
		}
		double dim = *std::max_element(gridDimensions.begin(), gridDimensions.end()) + 2*sensorSpacerX;
		gridCenterX[x] = runningDimX + dim/2.0;
		runningDimX += dim;
	}

	for(int y = 0; y < nBinsY; y++) {
		std::vector<double> gridDimensions;
		for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {
			if(gridIndexY[m] == y) {
				gridDimensions.push_back(dimensionY[m]);
			}
		}
		double dim = *std::max_element(gridDimensions.begin(), gridDimensions.end()) + 2*sensorSpacerY;
		gridCenterY[y] = runningDimY + dim/2.0;
		runningDimY += dim;
	}

	for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {
		m_newCenterX[m] = gridCenterX[gridIndexX[m]];
		m_newCenterY[m] = gridCenterY[gridIndexY[m]];
	}

	// Calculate sensor cell positions: vector overkill ;-)
	m_matrixCellCenterX.resize(m_frameManager->getNumCells());
	m_matrixCellCenterY.resize(m_frameManager->getNumCells());
	m_rectangleTopLeftX.resize(m_frameManager->getNumCells());
	m_rectangleTopLeftY.resize(m_frameManager->getNumCells());
	m_rectangleBottomRightX.resize(m_frameManager->getNumCells());
	m_rectangleBottomRightY.resize(m_frameManager->getNumCells());
	m_rectangleWidth.resize(m_frameManager->getNumCells());
	m_rectangleHeight.resize(m_frameManager->getNumCells());

	for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {

		matrixInfo &matrix = m_frameManager->getMatrixInfo(m);

		double topLeftTexelCenterX = m_newCenterX[m] - matrix.cells_x/2.0 * matrix.texel_width  + matrix.texel_width/2;
		double topLeftTexelCenterY = m_newCenterY[m] - matrix.cells_y/2.0 * matrix.texel_height + matrix.texel_height/2;

		double cellWidth = matrix.texel_width - matrix.texel_width/5;
		double cellHeight = matrix.texel_height  - matrix.texel_height/5;

		double cellCenterX = topLeftTexelCenterX;
		double cellCenterY = topLeftTexelCenterY;

		for(uint y = 0; y < matrix.cells_y; y++) {
			cellCenterX = topLeftTexelCenterX;
			for(uint x = 0; x < matrix.cells_x; x++) {
				uint cellID = matrix.texel_offset + y * matrix.cells_x + x;

				m_matrixCellCenterX[cellID] = cellCenterX;
				m_matrixCellCenterY[cellID] = cellCenterY;
				m_rectangleTopLeftX[cellID] = cellCenterX - cellWidth/2;
				m_rectangleTopLeftY[cellID] = cellCenterY - cellHeight/2;
				m_rectangleWidth[cellID] = cellWidth;
				m_rectangleHeight[cellID] = cellHeight;
				m_rectangleBottomRightX[cellID] = m_rectangleTopLeftX[cellID] + m_rectangleWidth[cellID];
				m_rectangleBottomRightY[cellID] = m_rectangleTopLeftY[cellID] + m_rectangleHeight[cellID];

				cellCenterX += matrix.texel_width;
			}
			cellCenterY += matrix.texel_height;
		}
	}

	// Calculate bounding box of matrices
	m_leftMargin = *std::min_element(m_rectangleTopLeftX.begin(), m_rectangleTopLeftX.end()) - sensorSpacerX;
	m_rightMargin = *std::max_element(m_rectangleBottomRightX.begin(), m_rectangleBottomRightX.end()) + sensorSpacerX;
	m_topMargin = *std::min_element(m_rectangleTopLeftY.begin(), m_rectangleTopLeftY.end()) - sensorSpacerY;
	m_bottomMargin = *std::max_element(m_rectangleBottomRightY.begin(), m_rectangleBottomRightY.end()) + sensorSpacerY;

	m_sensorWidth = m_rightMargin - m_leftMargin;
	m_sensorHeight = m_bottomMargin - m_topMargin;

	m_layoutAvailable = true;
}


bool guiRenderer2D::on_expose_event(GdkEventExpose* event) {

	Glib::RefPtr<Gdk::Window> window = get_window();
	if(window) {
		Cairo::RefPtr<Cairo::Context> cr = window->create_cairo_context();
		if(event) {
			// clip to the area indicated by the expose event so that we only
			// redraw the portion of the window that needs to be redrawn
			//printf("event->area.x: %d, event->area.y: %d, event->area.width: %d, event->area.height: %d\n", event->area.x, event->area.y, event->area.width, event->area.height );
			cr->rectangle(event->area.x, event->area.y,	event->area.width, event->area.height);
			cr->clip();
		}

		// Background
		// cr->set_source_rgb(0.0, 0.0, 0.0);
		cr->set_source_rgb(1.0, 1.0, 1.0);
		cr->paint();

		if(m_isRendering && m_layoutAvailable) {
			Gtk::Allocation allocation = get_allocation();
			int width = allocation.get_width();
			int height = allocation.get_height();

			if(width != m_widgetWidth || height != m_widgetHeight ) { // Allocation changed
				rescaleSensorLayout(width, height);
			}

			drawMatrices(cr, width, height, false);
		}
	}

	return true;
}


void guiRenderer2D::rescaleSensorLayout(int width, int height) {

	// Calculate scale and translation
	m_widgetWidth = width;
	m_widgetHeight = height;
	double ratioWidth = m_widgetWidth / m_sensorWidth;
	double ratioHeight = m_widgetHeight / m_sensorHeight;
	double centerOffsetX;
	double centerOffsetY;
	if(ratioWidth <= ratioHeight) {
		m_scaleFactor = ratioWidth;
		centerOffsetX = 0.0;
		double heightDifference = m_widgetHeight - m_scaleFactor * m_sensorHeight;
		centerOffsetY = (heightDifference/2.0) / m_scaleFactor;
	} else {
		m_scaleFactor = ratioHeight;
		double widthDifference = m_widgetWidth - m_scaleFactor * m_sensorWidth;
		centerOffsetX = (widthDifference/2.0) / m_scaleFactor;
		centerOffsetY = 0.0;
	}

	m_offsetX = -m_leftMargin + centerOffsetX;
	m_offsetY = -m_topMargin + centerOffsetY;
}


void guiRenderer2D::drawMatrices(const Cairo::RefPtr<Cairo::Context>& cr, int width, int height, bool screenshot) {

	cr->scale(m_scaleFactor, m_scaleFactor); // Scale sensor to fit the active window
	cr->translate(m_offsetX, m_offsetY); // Center figure on drawable/surface

	cr->set_line_width(0.25);

	for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {
		matrixInfo &matrix = m_frameManager->getMatrixInfo(m);
		// TSFrame* tsFrame = m_frameManager->getCurrentFrame();
		TSFrame* tsFrame = m_frameManager->getCurrentFilteredFrame();

		for(uint y = 0; y < matrix.cells_y; y++) {
			for(uint x = 0; x < matrix.cells_x; x++) {

				bool maskedStatic = m_frameManager->getStaticMask(m, x, y);
				bool maskedDynamic = m_frameManager->getDynamicMask(m, x, y);
				uint cellID = matrix.texel_offset + y * matrix.cells_x + x;
				float value = tsFrame->cells[cellID];

				if(maskedStatic) {
					RGB color = determineColor(value);

					// Draw sensor cell rectangle
					cr->rectangle(m_rectangleTopLeftX[cellID], m_rectangleTopLeftY[cellID], m_rectangleWidth[cellID], m_rectangleHeight[cellID]);
					cr->set_source_rgb(0.0, 0.0, 0.0);
					cr->stroke_preserve(); // Cell outline

					if(maskedDynamic) {
						if(value > 0.0) {
							cr->set_source_rgb(color.r, color.g, color.b); // Active cells
						} else  {
							cr->set_source_rgb(1.0, 1.0, 1.0); // Inactive cells
						}
					} else {
						cr->set_source_rgb(0.8, 0.8, 0.8); // Disabled cells
					}
					cr->fill();
				}

				// Highlight selected cells
				if(m_frameManager->isSelected(cellID)) {
					cr->rectangle(m_rectangleTopLeftX[cellID], m_rectangleTopLeftY[cellID], m_rectangleWidth[cellID], m_rectangleHeight[cellID]);
					cr->set_source_rgba(0.0, 1.0, 0.0, 0.5); // Fill active cells
					cr->fill();
				}

				if(screenshot) {
					if(maskedStatic) {
						// Print values
						Cairo::RefPtr<Cairo::ToyFontFace> font = Cairo::ToyFontFace::create("LMSans10", Cairo::FONT_SLANT_NORMAL, Cairo::FONT_WEIGHT_NORMAL);
						cr->set_font_face(font);
						cr->set_font_size(matrix.texel_width/3);
						std::ostringstream ss;
						ss << value;
						std::string valueStr = ss.str();

						Cairo::TextExtents te;
						cr->get_text_extents(valueStr, te);

						cr->move_to(m_matrixCellCenterX[cellID]-te.width/2, m_matrixCellCenterY[cellID]+te.height/2);
						cr->set_source_rgb(0.0, 0.0, 0.0);
						cr->show_text(valueStr);
					}
				}

			}
		}

		if(!screenshot) {
			{
				// Print Matrix IDs
				Cairo::RefPtr<Cairo::ToyFontFace> font = Cairo::ToyFontFace::create("LMSans10", Cairo::FONT_SLANT_NORMAL, Cairo::FONT_WEIGHT_NORMAL);
				cr->set_font_face(font);
				cr->set_font_size(matrix.cells_x*matrix.texel_width);
				std::ostringstream ss;
				ss << m;
				std::string idString = ss.str();

				Cairo::TextExtents te;
				cr->get_text_extents(idString, te);

				cr->move_to(m_newCenterX[m]-te.width/2, m_newCenterY[m]+te.height/2);
				cr->set_source_rgba(0.3, 0.3, 0.3, 0.3);
				cr->show_text(idString);
			}
		}

	}
}


void guiRenderer2D::takeScreenshot(const string& filename) {
	// PDF document size in points
	int pdfWidth = 500;
	int pdfHeight = 500;

	// Save window size
	int tmpWidgetWidth = m_widgetWidth;
	int tmpWidgetHeight = m_widgetHeight;

	rescaleSensorLayout(pdfWidth, pdfHeight);

	Cairo::RefPtr<Cairo::PdfSurface> surface = Cairo::PdfSurface::create(filename, pdfWidth, pdfHeight);
	Cairo::RefPtr<Cairo::Context> cr = Cairo::Context::create(surface);
	drawMatrices(cr, pdfWidth, pdfHeight, true);
	cr->show_page();

	// Restore window size
	m_widgetWidth = tmpWidgetWidth;
	m_widgetHeight = tmpWidgetHeight;
	rescaleSensorLayout(m_widgetWidth, m_widgetWidth);
}


// Invalidate the whole window
// Forces widget to be redrawn in the near future
void guiRenderer2D::invalidate() {
	Glib::RefPtr<Gdk::Window> window = get_window();
	if(window) {
		//get_window()->invalidate_rect(get_allocation(), false);
		get_window()->invalidate(false);
	}
}


// Update window synchronously (fast)
// Causes the redraw to be done immediately
void guiRenderer2D::update() {
	Glib::RefPtr<Gdk::Window> window = get_window();
	if(window) {
		get_window()->process_updates(false);
	}

}


void guiRenderer2D::renderFrame() {
	this->queue_draw();
}


void guiRenderer2D::renderFrame(uint frameID) {
	m_frameManager->setCurrentFrameID(frameID);
	this->queue_draw();
}


void guiRenderer2D::determineSelection(int x, int y) {
	uint selectedCellsBefore = m_frameManager->getNumSelectedCells();
	double clickedX = x / m_scaleFactor - m_offsetX;
	double clickedY = (m_widgetHeight - y) / m_scaleFactor - m_offsetY;

	uint cellID;
	bool cellSelected = false;

	// Calculate intersection with sensor cells
	for(uint i = 0; i < m_frameManager->getNumCells(); i++) {
		if( clickedX > m_rectangleTopLeftX[i] &&
				clickedX < m_rectangleBottomRightX[i] &&
				clickedY > m_rectangleTopLeftY[i] &&
				clickedY < m_rectangleBottomRightY[i] ) {
			printf("Cell %d selected!\n", i);
			cellID = i;
			cellSelected = true;
			break;
		}
	}

	if(cellSelected) {
		// Toggle selection
		if(m_mouseDragging) { // Mark new cell with the same value as the previous one
			m_frameManager->selectCell(cellID, m_previousSelectionState);
		} else { // Single click (toggle state)
			m_frameManager->isSelected(cellID) ? m_frameManager->selectCell(cellID, false) : m_frameManager->selectCell(cellID, true);
			m_previousSelectionState = m_frameManager->isSelected(cellID);
		}
	}
	uint selectedCellsAfter = m_frameManager->getNumSelectedCells();

	if(m_mainGUI) {
		if(selectedCellsBefore != selectedCellsAfter) {
			m_mainGUI->updateDataset();
		}
	}
}


// Mouse button pressed
bool guiRenderer2D::on_button_press_event(GdkEventButton* event) {
	// Left mouse button
	if(event->button == 1) {
		m_mouseLeftButtonDown = true;
		m_mouseLeftPressed.x() = event->x; // Initialize the starting coordinates for further mouse movement
		m_mouseLeftPressed.y() = get_height()-event->y; // Invert y-axis

		if(m_selectionMode) {
			determineSelection(m_mouseLeftPressed.x(), m_mouseLeftPressed.y());
			invalidate();
		}
	}

	// Right mouse button
	if(event->button == 3) {
		m_mouseRightButtonDown = true;
		m_mouseRightPressed.x() = event->x; // Initialize the starting coordinates for further mouse movement
		m_mouseRightPressed.y() = get_height()-event->y; // Invert y-axis
		// Show popup menu
		m_Menu_Popup.popup(event->button, event->time);
	}

	return true;
}



// Mouse button released
bool guiRenderer2D::on_button_release_event(GdkEventButton* event) {
	// Left button
	if(event->button == 1) {
		m_mouseLeftButtonDown = false;
	}

	// Right button
	if(event->button == 3) {
		m_mouseRightButtonDown = false;
	}

	m_mouseDragging = false;
	return true;
}


// Moving the mouse with pressed buttons
bool guiRenderer2D::on_motion_notify_event(GdkEventMotion* event) {
	if(m_selectionMode) {
		if(m_mouseLeftButtonDown) {
			m_mouseDragging = true;
			determineSelection(event->x, get_height()-event->y); // Invert y-axis
			invalidate();
		}
	}

	return true;
}


bool guiRenderer2D::on_key_press_event(GdkEventKey* event) {

	// Selection Mode
	if(event->keyval == GDK_KEY_Control_L) {
		m_selectionMode = true;
	}
	return true; // Prevent further propagation to focused child-widget
}


bool guiRenderer2D::on_key_release_event(GdkEventKey* event) {

	// Selection Mode
	if(event->keyval == GDK_KEY_Control_L) {
		m_selectionMode = false;
	}
	return true; // Prevent further propagation to focused child-widget
}


void guiRenderer2D::on_menu_popup_set_mask() {
	if(m_frameManager->isConnectedDSA()) {
		m_frameManager->setDynamicMask(m_frameManager->getSelection());
	} else {
		Gtk::MessageDialog dialog("DSA is not connected!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
		dialog.run();
	}
}


void guiRenderer2D::on_menu_popup_reset_mask() {
	if(m_frameManager->isConnectedDSA()) {
		std::vector<bool> all_true(m_frameManager->getNumCells(), true);
		m_frameManager->setDynamicMask(all_true);
	} else {
		Gtk::MessageDialog dialog("DSA is not connected!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
		dialog.run();
	}
}
