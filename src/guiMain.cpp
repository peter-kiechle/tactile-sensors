#include "guiMain.h"

#include <numpy/ndarrayobject.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/median.hpp>

guiMain::guiMain(Controller* c) : m_featureExtractor(*(c->getFrameManager())) {
	m_controller = c;
	m_frameManager = m_controller->getFrameManager();
	m_frameProcessor = m_frameManager->getFrameProcessor();
	current_frame = 0;

	set_title("DSA-Konqueror");
	set_icon_from_file("icons/DSAKonqueror.svg");
	set_border_width(0);
	set_default_size(1024, 736);
	//maximize();

	// Catch resize event
	signal_configure_event().connect_notify(sigc::mem_fun(*this, &guiMain::on_resize_notify) );
	m_resized = true;

	showSensorView = true;
	showChartView = false;
	showTreeView = false;

	m_pythonEmbedded = false;

	// Menubar
	m_VBox_Main.pack_start(m_Menubar, Gtk::PACK_SHRINK);
	Gtk::MenuItem* pMenuItem = Gtk::manage(new Gtk::MenuItem("File"));
	m_Menubar.append(*pMenuItem);

	Gtk::Menu* menu = Gtk::manage(new Gtk::Menu());
	pMenuItem->set_submenu(*menu);

	Gtk::ImageMenuItem *pImageMenuItem = Gtk::manage(new Gtk::ImageMenuItem(Gtk::Stock::OPEN));
	pImageMenuItem->set_label("Open Pressure Profile");
	pImageMenuItem->signal_activate().connect( sigc::mem_fun(*this, &guiMain::on_menu_load_profile) );
	menu->append(*pImageMenuItem);

	menu->append(*Gtk::manage(new Gtk::SeparatorMenuItem()));

	pImageMenuItem = Gtk::manage(new Gtk::ImageMenuItem(Gtk::Stock::PRINT_PREVIEW));
	pImageMenuItem->set_label("Export current frame as PDF (2D)");
	pImageMenuItem->signal_activate().connect( sigc::mem_fun(*this, &guiMain::on_menu_take_screenshot_2D_clicked) );
	menu->append(*pImageMenuItem);

	pImageMenuItem = Gtk::manage(new Gtk::ImageMenuItem(Gtk::Stock::PRINT_REPORT));
	pImageMenuItem->set_label("Export frame(s) as PNG (3D)");
	pImageMenuItem->signal_activate().connect( sigc::mem_fun(*this, &guiMain::on_menu_take_screenshot_3D_clicked) );
	menu->append(*pImageMenuItem);

	menu->append(*Gtk::manage(new Gtk::SeparatorMenuItem()));

	pImageMenuItem = Gtk::manage(new Gtk::ImageMenuItem(Gtk::Stock::QUIT));
	pImageMenuItem->signal_activate().connect( sigc::mem_fun(*this, &guiMain::on_menu_file_quit) );
	menu->append(*pImageMenuItem);

	// Toolbar
	m_Toolbar.set_toolbar_style(Gtk::TOOLBAR_BOTH); // Icons and labels
	m_VBox_Main.pack_start(m_Toolbar, Gtk::PACK_SHRINK);

	// Toolbar buttons
	Gtk::ToolButton* toolbutton = Gtk::manage(new Gtk::ToolButton(Gtk::Stock::NEW));
	toolbutton->set_label("New Profile");
	m_Toolbar.append(*toolbutton, sigc::mem_fun(*this, &guiMain::on_menu_new_profile));

	toolbutton = Gtk::manage(new Gtk::ToolButton(Gtk::Stock::OPEN));
	toolbutton->set_label("Load Profile");
	m_Toolbar.append(*toolbutton, sigc::mem_fun(*this, &guiMain::on_menu_load_profile));

	toolbutton = Gtk::manage(new Gtk::ToolButton(Gtk::Stock::SAVE_AS));
	toolbutton->set_label("Save Profile");
	m_Toolbar.append(*toolbutton, sigc::mem_fun(*this, &guiMain::on_menu_save_profile_as));

	m_Toolbar.append(*Gtk::manage(new Gtk::SeparatorToolItem()));

	m_ToggleToolButton_Connect_SDH.set_label("SDH");
	m_ToggleToolButton_Connect_SDH.set_stock_id(Gtk::Stock::DISCONNECT);
	m_ToggleToolButton_Connect_SDH_pressed = false;
	m_Toolbar.append(m_ToggleToolButton_Connect_SDH, sigc::mem_fun(*this, &guiMain::on_menu_connect_SDH));

	m_ToggleToolButton_Connect_DSA.set_label("DSA");
	m_ToggleToolButton_Connect_DSA.set_stock_id(Gtk::Stock::DISCONNECT);
	m_ToggleToolButton_Connect_DSA_pressed = false;
	m_Toolbar.append(m_ToggleToolButton_Connect_DSA, sigc::mem_fun(*this, &guiMain::on_menu_connect_DSA));

	m_Toolbar.append(*Gtk::manage(new Gtk::SeparatorToolItem()));

	m_ToggleToolButton_Sensor_View.set_label("Sensor Matrix");
	m_ToggleToolButton_Sensor_View.set_icon_widget(*Gtk::manage(new Gtk::Image("icons/DSAKonqueror.svg")) );
	m_ToggleToolButton_Sensor_View.set_active(showSensorView);
	m_Toolbar.append(m_ToggleToolButton_Sensor_View, sigc::mem_fun(*this, &guiMain::on_menu_show_sensor_view));

	m_ToggleToolButton_Chart_View.set_label("Chart");
	m_ToggleToolButton_Chart_View.set_icon_widget(*Gtk::manage(new Gtk::Image("icons/office-chart-line-stacked.svg")) );
	m_ToggleToolButton_Chart_View.set_active(showChartView);
	m_Toolbar.append(m_ToggleToolButton_Chart_View, sigc::mem_fun(*this, &guiMain::on_menu_show_chart_view));

	m_ToggleToolButton_Tree_View.set_label("Matrix Info");
	m_ToggleToolButton_Tree_View.set_stock_id(Gtk::Stock::INFO);
	m_ToggleToolButton_Tree_View.set_active(showTreeView);
	m_Toolbar.append(m_ToggleToolButton_Tree_View, sigc::mem_fun(*this, &guiMain::on_menu_show_tree_view));

	m_ToggleToolButton_Tools.set_label("OpenCV");
	m_ToggleToolButton_Tools.set_icon_widget(*Gtk::manage(new Gtk::Image("icons/opencv.svg")) );
	m_Toolbar.append(m_ToggleToolButton_Tools, sigc::mem_fun(*this, &guiMain::on_menu_show_tools));

	m_ToggleToolButton_Slip_Detection.set_label("Slip Detection");
	m_ToggleToolButton_Slip_Detection.set_icon_widget(*Gtk::manage(new Gtk::Image("icons/slip-detection.svg")) );
	m_Toolbar.append(m_ToggleToolButton_Slip_Detection, sigc::mem_fun(*this, &guiMain::on_menu_show_slip_detection));


	toolbutton = Gtk::manage(new Gtk::ToolButton(Gtk::Stock::NEW));
	toolbutton->set_label("Classifier");
	toolbutton->set_icon_widget(*Gtk::manage(new Gtk::Image("icons/classification.svg")) );
	m_Toolbar.append(*toolbutton, sigc::mem_fun(*this, &guiMain::on_menu_classify));


	m_Frame_Controller = Gtk::manage(new guiController(m_controller));
	m_Frame_Controller->set_sensitive(true);
	m_HBox_Main.pack_start(*m_Frame_Controller, Gtk::PACK_SHRINK, 5);

	// Cairo based 2D renderer
	m_guiRenderer2D = Gtk::manage(new guiRenderer2D(m_controller->getFrameManager(), this));
	m_Frame_Renderer2D.add(*m_guiRenderer2D);
	m_Frame_Renderer2D.set_shadow_type(Gtk::SHADOW_IN);

	// OpenGL based 3D renderer
	m_guiRenderer3D = Gtk::manage(new guiRenderer3D(m_controller->getFrameManager(), this));
	m_Frame_Renderer3D.add(*m_guiRenderer3D);
	m_Frame_Renderer3D.set_shadow_type(Gtk::SHADOW_IN);

	renderer = RENDERER_OPENGL;
	m_controller->setRenderer(m_guiRenderer3D);

	// Renderer Notebook (Tabs)
	m_Notebook_Renderer.append_page(m_Frame_Renderer3D, "OpenGL");
	m_Notebook_Renderer.append_page(m_Frame_Renderer2D, "Cairo");
	m_Notebook_Renderer.signal_switch_page().connect(sigc::mem_fun(*this, &guiMain::on_notebook_switch_page) );
	m_VBox_Renderer.pack_start(m_Notebook_Renderer, Gtk::PACK_EXPAND_WIDGET, 0);

	m_guiScreenshot = NULL;

	// Seekbar
	m_guiSeekbar = Gtk::manage(new guiSeekbar(m_controller, this));
	m_VBox_Renderer.pack_start(*m_guiSeekbar, Gtk::PACK_SHRINK, 5);

	// VPaned (Renderer/Chart)
	if(showSensorView) {
		m_VPaned_Views.add1(m_VBox_Renderer);
	}
	m_guiChart = Gtk::manage(new guiChart(m_controller, this));
	m_guiChart->set_sensitive(false);
	if(showChartView) {
		m_VPaned_Views.add2(*m_guiChart);
	}
	m_VPaned_Views_Ratio = 0.8;
	m_VPaned_Views.signal_size_allocate().connect(sigc::mem_fun(*this, &guiMain::on_vpaned_size_allocate));
	m_VPaned_Views.signal_realize().connect(sigc::mem_fun(*this, &guiMain::on_vpaned_realize));

	m_HBox_Main.pack_start(m_VPaned_Views, Gtk::PACK_EXPAND_WIDGET, 5);


	// Right sidebar
	m_guiTreeView = Gtk::manage(new guiTreeView(m_controller, this));

	m_VBox_Right_Sidebar.pack_start(*m_guiTreeView, Gtk::PACK_EXPAND_WIDGET, 0);
	m_HBox_Main.pack_start(m_VBox_Right_Sidebar, Gtk::PACK_SHRINK, 5);
	m_HBox_Main.set_border_width(3);

	m_VBox_Main.pack_start(m_HBox_Main, Gtk::PACK_EXPAND_WIDGET, 5);
	add(m_VBox_Main);

	// Tool window
	m_guiTools = NULL;

	// Slip detection window
	m_guiSlipDetection = NULL;

	show_all();

	m_VBox_Right_Sidebar.set_visible(showTreeView);

	resetGUIOffline();
	if(m_controller->getFrameManager()->getTSFrameAvailable()) {
		updateGUIOffline();
	}
}

guiMain::~guiMain() {
	delete m_guiTools;
	delete m_guiSlipDetection;

	if(m_pythonEmbedded) {
		Py_Finalize();
	}
}


bool guiMain::on_key_press_event(GdkEventKey* event) {
	return m_controller->getRenderer()->on_key_press_event(event);
}


bool guiMain::on_key_release_event(GdkEventKey* event) {
	return m_controller->getRenderer()->on_key_release_event(event);
}


void guiMain::on_menu_file_quit() {
	hide(); // Closes the main window to stop the Gtk::Main::run().
}


void guiMain::on_menu_connect_SDH() {

	if(m_controller->isConnectedSDH()) { // Disconnect
		m_ToggleToolButton_Connect_SDH.set_stock_id(Gtk::Stock::DISCONNECT);
		m_Frame_Controller->disconnectSDH();
		m_ToggleToolButton_Connect_SDH_pressed = false;
	} else { // Connect
		if(m_ToggleToolButton_Connect_SDH_pressed == false) {
			m_ToggleToolButton_Connect_SDH_pressed = true;
			m_Frame_Controller->connectSDH();
			if(m_controller->isConnectedSDH()) {
				m_ToggleToolButton_Connect_SDH.set_stock_id(Gtk::Stock::CONNECT);
			} else {
				// Despite not being connected, the toggle button's state is "pressed in"
				m_ToggleToolButton_Connect_SDH.set_active(false); // Signal is emitted, i.e. recursive call of on_menu_connect_SDH
			}
		} else {
			m_ToggleToolButton_Connect_SDH_pressed = false;
		}
	}
}


void guiMain::on_menu_connect_DSA() {

	if(m_controller->isConnectedDSA()) { // Disconnect
		if(m_guiSlipDetection != NULL) {
			m_guiSlipDetection->setModeOffline();
		}
		m_ToggleToolButton_Connect_DSA.set_stock_id(Gtk::Stock::DISCONNECT);
		m_Frame_Controller->disconnectDSA();
		m_ToggleToolButton_Connect_DSA_pressed = false;
	} else { // Connect
		if(m_ToggleToolButton_Connect_DSA_pressed == false) {
			m_ToggleToolButton_Connect_DSA_pressed = true;
			m_Frame_Controller->connectDSA();
			if(m_controller->isConnectedDSA()) {
				m_ToggleToolButton_Connect_DSA.set_stock_id(Gtk::Stock::CONNECT);
				resetGUIOnline();
				if(m_guiSlipDetection != NULL) {
					m_guiSlipDetection->setModeOnline();
				}
			} else {
				// Despite not being connected, the toggle button's state is "pressed in"
				m_ToggleToolButton_Connect_DSA.set_active(false); // Signal is emitted, i.e. recursive call of on_menu_connect_DSA
			}
		} else {
			m_ToggleToolButton_Connect_DSA_pressed = false;
		}
	}
}


/**
 * Reset FrameManager
 */
void guiMain::on_menu_new_profile() {

	if( m_controller->isAvailableSDH() || m_controller->isAvailableDSA() ) {
		m_frameManager->resetOnline();
		resetGUIOnline();
	} else {
		Gtk::MessageDialog dialog("No device connected!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
		dialog.run();
	}
}

/**
 * File chooser for *.dsa pressure profile file
 */
void guiMain::on_menu_load_profile() {

	Gtk::FileChooserDialog dialog("Choose Pressure Profile", Gtk::FILE_CHOOSER_ACTION_OPEN);
	dialog.set_transient_for(*this);
	dialog.set_current_folder(boost::filesystem::current_path().string());

	//Add response buttons the the dialog:
	dialog.add_button(Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL);
	dialog.add_button(Gtk::Stock::OPEN, Gtk::RESPONSE_OK);

	// Add filters dsa file types
	Gtk::FileFilter filter_dsa;
	filter_dsa.set_name("Pressure profiles");
	filter_dsa.add_mime_type("application/x-bzip2");
	filter_dsa.add_pattern("*.dsa");
	dialog.add_filter(filter_dsa);

	Gtk::FileFilter filter_any;
	filter_any.set_name("All files");
	filter_any.add_pattern("*");
	dialog.add_filter(filter_any);

	// Show the dialog and wait for a user response:
	int result = dialog.run();

	// Handle the response
	switch(result) {
	case(Gtk::RESPONSE_OK):
			  {
		//Notice that this is a std::string, not a Glib::ustring.
		string filename = dialog.get_filename();

		resetGUIOffline();
		m_controller->loadProfile(filename);
		updateGUIOffline();
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


void guiMain::on_menu_save_profile_as() {

	Gtk::FileChooserDialog dialog("Save Pressure Profile", Gtk::FILE_CHOOSER_ACTION_SAVE);
	dialog.set_transient_for(*this);
	dialog.set_current_folder(boost::filesystem::current_path().string());

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

		fprintf(stderr, "Filename: %s\n", filename.c_str());

		m_frameManager->storeFrames(filename);

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


void guiMain::on_menu_show_tools() {
	if(m_guiTools == NULL) {
		m_guiTools = new guiTools(m_controller, this); // Gtk::manage cannot be used for Gtk::Window
		m_guiTools->signal_delete_event().connect(sigc::mem_fun(*this, &guiMain::on_tools_delete_clicked));
		m_guiTools->show_all();
	} else {
		if(!m_guiTools->get_visible()) {
			m_guiTools->set_visible(true);
		} else {
			m_guiTools->set_visible(false);
		}
	}
}


void guiMain::on_menu_show_slip_detection() {
	if(m_guiSlipDetection == NULL) {
		m_guiSlipDetection = new guiSlipDetection(m_controller, this); // Gtk::Window cannot be managed
		m_guiSlipDetection->signal_delete_event().connect(sigc::mem_fun(*this, &guiMain::on_slip_detection_delete_clicked));
		m_guiSlipDetection->show_all();
	} else {
		if(!m_guiSlipDetection->get_visible()) {
			m_guiSlipDetection->set_visible(true);
		} else {
			m_guiSlipDetection->set_visible(false);
		}
	}
}


/**
 * Embed python interpreter, load trained state of SVM and classify feature vector
 */
void guiMain::embedPython() {

	if(!m_pythonEmbedded) {

		try {
			Py_Initialize(); // Initialize python interpreter

			// Retrieve python's main module
			m_main = bp::import("__main__");

			// Retrieve the main module's namespace
			m_global = m_main.attr("__dict__");

			// Execute python script
			bp::object python_script = exec_file("/home/gorth/masterthesis/boost-python-cmake/sdh_tactile_sensor/utilities/svm/predict_boost_python.py", m_global, m_global);

			// Needed for conversion from std::vector to numpy array
			// http://stackoverflow.com/questions/10701514/how-to-return-numpy-array-from-boostpython
			import_array(); // NumPy function
			bp::numeric::array::set_module_and_type("numpy", "ndarray");

		} catch( const bp::error_already_set& e ) {
			PyErr_Print();
		}
		m_pythonEmbedded = true;
	}
}


void guiMain::on_menu_classify() {

	printf("\nClassify\n");

	// Determine current connection state
	enum connectionMode { OFFLINE, ONLINE, INACTIVE };
	connectionMode mode;
	if(m_frameManager->isConnectedDSA()) {
		mode = ONLINE;
	} else if( m_frameManager->getTSFrameAvailable() && m_frameManager->getJointAngleFrameAvailable() ) {
		mode = OFFLINE;
	} else {
		mode = INACTIVE;
	}

	// Grasp feature properties
	int pmax = 5; // Maximum moment order
	int headWindow = 10; // Search window size for initial position
	int tailWindow = 10; // Search window size for end position
	int threshSequence = 10; // Minimum number of empty frames to safely assume grasp separation
	int startFrame;
	int stopFrame;
	int range;
	double max_val_matrix_1 = 3554.0;
	double max_val_matrix_5 = 2493.0;
	double impression_depth = 1.0; // Just an estimate of the maximal impression in [mm]
	double impression_factor_1 = impression_depth / max_val_matrix_1;
	double impression_factor_5 = impression_depth / max_val_matrix_5;

	int n_features = 1 + 1 + 2*1 + 2*pmax*pmax; // Feature vector size
	std::vector<double> featureVector;
	int frameID_initial_position;
	int frameID_end_position;
	bool isGrabbingDSA = false;
	bool isGrabbingSDH = false;

	if(mode == OFFLINE) {

		// Sanity checks
		if(getActiveSelection()) {
			startFrame = getSelectionFrom();
			stopFrame = getSelectionTo();
			range = stopFrame - startFrame + 1;

			if(range > 5000) {
				Gtk::MessageDialog dialog("Selection is too large (Should be < 5000 frames)", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
				dialog.run();
				return;
			}
			//printf("Selection: [%d, %d]\n", startFrame, stopFrame);
		} else {
			Gtk::MessageDialog dialog("Selection is empty!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
			dialog.run();
			return;
		}

	} else if(mode == ONLINE) {

		// Sanity checks
		if( m_frameManager->getTSFrameAvailable() && m_frameManager->getJointAngleFrameAvailable() ) {

			// Pause grabbing
			if( m_controller->getFrameGrabberDSA()->isCapturing() && m_controller->getFrameGrabberDSA()->isRecording() ) {
				isGrabbingDSA = true;
			} else {
				Gtk::MessageDialog dialog("DSA frame grabber is not recording", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
				dialog.run();
				return;
			}

			if(m_controller->getFrameGrabberSDH()->isCapturing() && m_controller->getFrameGrabberSDH()->isRecording() ) {
				isGrabbingSDH = true;
			} else {
				Gtk::MessageDialog dialog("SDH frame grabber is not recording", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
				dialog.run();
				return;
			}

		} else {
			Gtk::MessageDialog dialog("Error not recording!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
			dialog.run();
			return;
		}

		if(isGrabbingDSA && isGrabbingSDH) {
			m_controller->getFrameGrabberDSA()->pause();
			m_controller->getFrameGrabberSDH()->pauseBlocking();
			stopFrame = m_frameManager->getFrameCountTS()-1;
			startFrame = std::max(stopFrame-5000, 0);
			range = stopFrame - startFrame + 1;
			m_frameManager->createJointAngleMapping();
		}
	}

	// Determine frames of interest

	if(mode == ONLINE || mode == OFFLINE) {

		//printf("startFrame: %d, stopFrame: %d, range: %d\n", startFrame, stopFrame, range);

		float maxMatrix_1 = m_frameProcessor->getMatrixMax(stopFrame, 1);
		float maxMatrix_5 = m_frameProcessor->getMatrixMax(stopFrame, 5);
		if( !(maxMatrix_1 > 0.0 && maxMatrix_5 > 0.0) ) {
			Gtk::MessageDialog dialog("Last frame has to be under load", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
			dialog.run();
			goto Cleanup;
		}

		// Compute minimal bounding sphere for entire range
		JointAngleFrame *jointAngleFrame;
		std::vector<double> mbsDiameter(range, 0.0);
		for(int frameID = startFrame, i = 0; frameID <= stopFrame; frameID++, i++) {
			maxMatrix_1 = m_frameProcessor->getMatrixMax(frameID, 1);
			maxMatrix_5 = m_frameProcessor->getMatrixMax(frameID, 5);
			if(maxMatrix_1 > 0.0 && maxMatrix_5 > 0.0) {
				jointAngleFrame = m_frameManager->getCorrespondingJointAngle(frameID);
				std::vector<double> miniball = m_featureExtractor.computeMiniballCentroid(frameID, jointAngleFrame->angles);
				// Compensate for force dependent sensor matrix impression
				mbsDiameter[i] = 2*miniball[3] + maxMatrix_1*impression_factor_1 + maxMatrix_5*impression_factor_5;
			}
		}


		// Compute radius median of *tailWindow* frames (more precisely: center element of sorted array in case array is odd)
		// Create a copy since nth_element() rearranges array
		std::vector<double>::iterator first = mbsDiameter.end()-tailWindow;
		std::vector<double>::iterator last = mbsDiameter.end();
		std::vector<double> tail(first, last);
		size_t n = tail.size()/2;
		std::nth_element( tail.begin(), tail.begin() + n, tail.end() );
		double median_tail = tail[n];

		// Find index of median in original array (search in tail only)
		std::vector<double>::iterator itMedian = std::find(first, last, median_tail);
		int endPositionRange = std::distance(mbsDiameter.begin(), itMedian);
		//printf("Median: %2.3f at frameID %d (%2.3f)\n", median, startFrame+endPositionRange, mbsDiameter[endPositionRange]);

		// Beginning at frameID_end_position, move backwards to find grasp start
		// i.e. a couple of consecutive frames with invalid minimal bounding spheres
		int invalidCounter = 0;
		int lastValidRange = endPositionRange;
		for(int i = endPositionRange; i >= 0; i--) {

			if( !(mbsDiameter[i] > 0.0) ) { // Invalid minimal bounding sphere
				invalidCounter++;
				//printf("FAIL: Frame: %d, mbsDiameter[%d] > 0.0 (%2.3f),   invalidCounter: %d\n", startFrame + i, i, mbsDiameter[i], invalidCounter);
			} else {
				invalidCounter = 0;
				lastValidRange = i;
				//printf("OK: Frame: %d, mbsDiameter[%d] > 0.0 (%2.3f),   invalidCounter: %d\n", startFrame + i, i, mbsDiameter[i], invalidCounter);
			}

			if(invalidCounter == threshSequence) { // Limit reached!
				break;
			}
		}
		//printf("endPositionRange: %d, invalidCounter: %d, lastValidRange: %d \n", endPositionRange, invalidCounter, lastValidRange);

		// Now take one of the first *headWindow* frames (the one with the largest miniball)
		std::vector<double>::iterator maxMbs;
		std::vector<double>::iterator itGraspStart = mbsDiameter.begin() + lastValidRange;
		std::vector<double>::iterator itInitialPosition;
		itInitialPosition = std::max_element(itGraspStart, itGraspStart + headWindow);
		int initialPositionRange = std::distance(mbsDiameter.begin(), itInitialPosition);

		frameID_initial_position = startFrame + initialPositionRange;
		frameID_end_position = startFrame + endPositionRange;


		// -----------------------
		// Compute feature vector
		// -----------------------
		printf("Computing feature vector: initial frame: %d (%d), last frame: %d (%d), Grasp length:  %d \n",
				frameID_initial_position, initialPositionRange, frameID_end_position, endPositionRange, frameID_end_position-frameID_initial_position);

		featureVector.resize(n_features);

		// Compute features
		double diameterInitial = mbsDiameter[initialPositionRange];
		double diameterEnd = mbsDiameter[endPositionRange];

		// Using boost accumulators...
		using namespace boost::accumulators;
		accumulator_set<double, stats<tag::median> > acc;
		acc = std::for_each( mbsDiameter.begin(), mbsDiameter.end(), acc );
		double diameter = median(acc);

		double compressibility = diameterInitial - diameterEnd;
		double std_dev_matrix_1 = m_featureExtractor.computeStandardDeviation(frameID_end_position, 1);
		double std_dev_matrix_5 = m_featureExtractor.computeStandardDeviation(frameID_end_position, 5);
		array_type moments_matrix_1 = m_featureExtractor.computeMoments(frameID_end_position, 1, pmax);
		array_type moments_matrix_5 = m_featureExtractor.computeMoments(frameID_end_position, 5, pmax);

		// Combine features
		featureVector[0] = diameter;
		featureVector[1] = compressibility;
		featureVector[2] = std_dev_matrix_1;
		featureVector[3] = std_dev_matrix_5;
		int offset = 4;
		for(int p = 0; p < pmax; p++) {
			for(int q = 0; q < pmax; q++) {
				featureVector[offset] = moments_matrix_1[p][q];
				offset++;
			}
		}
		for(int p = 0; p < pmax; p++) {
			for(int q = 0; q < pmax; q++) {
				featureVector[offset] = moments_matrix_5[p][q];
				offset++;
			}
		}


		// --------------------------------------------
		// Start trained Support Vector Machine (libsvm)
		// --------------------------------------------

		// Embed Python if not already done
		embedPython();

		try {
			// Conversion from std::vector to numpy array
			// http://stackoverflow.com/questions/10701514/how-to-return-numpy-array-from-boostpython
			npy_intp size = featureVector.size();
			double* data = size ? const_cast<double*>(&featureVector[0]) : static_cast<double*>(NULL); // Writable pointer
			PyObject *pyObj = PyArray_SimpleNewFromData( 1, &size, NPY_DOUBLE, data );
			bp::handle<> handle( pyObj );
			bp::numeric::array featureVectorNumpy(handle);
			featureVectorNumpy.copy(); // Let python own a copy

			bp::object predict = m_global["predict"];
			predict(featureVectorNumpy);

		} catch( const bp::error_already_set& e ) {
			PyErr_Print();
		}

	} else {
		Gtk::MessageDialog dialog("No recorded profile available!", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_OK, true);
		dialog.run();
		goto Cleanup;
	}


	Cleanup: // Yes that's right: spaghetti for lunch today

	if(isGrabbingDSA) {
		m_controller->getFrameGrabberDSA()->resume();
	}
	if(isGrabbingSDH) {
		m_controller->getFrameGrabberSDH()->resume();
	}

}


void guiMain::on_menu_show_sensor_view() {

	showSensorView = !showSensorView;
	if(showSensorView) {
		m_VPaned_Views.add1(m_VBox_Renderer);
	} else {
		m_VPaned_Views.remove(m_VBox_Renderer);
	}
	m_VPaned_Views.show_all();
}


void guiMain::on_menu_show_chart_view() {

	showChartView = !showChartView;
	if(showChartView) {
		m_VPaned_Views.add2(*m_guiChart);
	} else {
		m_VPaned_Views.remove(*m_guiChart);
	}
	m_VPaned_Views.show_all();
}


void guiMain::on_menu_show_tree_view() {
	showTreeView = !showTreeView;
	m_VBox_Right_Sidebar.set_visible(showTreeView);
}


bool guiMain::on_tools_delete_clicked(GdkEventAny* event) {
	m_ToggleToolButton_Tools.set_active(false);
	return true;
}

bool guiMain::on_slip_detection_delete_clicked(GdkEventAny* event) {
	m_ToggleToolButton_Slip_Detection.set_active(false);
	return true;
}


void guiMain::on_notebook_switch_page(GtkNotebookPage* page, guint page_num) {
	Glib::ustring tabLabel = m_Notebook_Renderer.get_tab_label_text(*m_Notebook_Renderer.get_nth_page(page_num));
	if(tabLabel.compare("Cairo") == 0) {
		m_controller->setRenderer(m_guiRenderer2D);
		renderer = RENDERER_CAIRO;
	} else if(tabLabel.compare("OpenGL") == 0) {
		m_controller->setRenderer(m_guiRenderer3D);
		renderer = RENDERER_OPENGL;
	}
}


void guiMain::on_menu_take_screenshot_2D_clicked() {
	saveCurrentFramePDF();
}


void guiMain::saveCurrentFramePDF() {
	if(m_frameManager->getTSFrameAvailable()) {
		std::string basename = m_controller->getProfileBaseName();
		std::ostringstream ss;
		ss << basename << "_frame_" << std::setfill('0') << std::setw(5) << m_frameManager->getCurrentFrameID() << ".pdf";
		std::string filename = ss.str();

		m_guiRenderer2D->takeScreenshot(filename);

		Gtk::MessageDialog dialog(*this, "Frame saved!", Gtk::MESSAGE_INFO);
		dialog.set_secondary_text(filename);
		dialog.run();
	} else {
		Gtk::MessageDialog dialog(*this, "No profile available", Gtk::MESSAGE_ERROR);
		dialog.run();
	}
}


void guiMain::on_menu_take_screenshot_3D_clicked() {
	if(m_guiScreenshot == NULL) {
		if(m_frameManager->getTSFrameAvailable()) {
			if(m_frameManager->isConnectedDSA()) { // Online
				// Create a filename from profile name and time stamp
				char timestamp[30];
				struct timeval tv;
				time_t curtime;
				gettimeofday(&tv, NULL);
				curtime = tv.tv_sec;
				strftime(timestamp, 30, "%d-%m-%Y_%T.", localtime(&curtime));
				std::string basename = m_controller->getProfileBaseName();
				std::ostringstream ss;
				ss << basename << "_frame_" << std::setfill('0') << std::setw(5) << m_frameManager->getCurrentFrameID() << "_" << timestamp << tv.tv_usec << ".png";
				std::string filename = ss.str();

				m_guiRenderer3D->takeScreenshot(filename);

				Gtk::MessageDialog dialog(*this, "Frame saved!", Gtk::MESSAGE_INFO);
				dialog.set_secondary_text(filename);
				dialog.run();

			} else { // Offline

				if(getActiveSelection()) {
					uint from = getSelectionFrom();
					uint to = getSelectionTo();
					m_guiScreenshot = new guiScreenshot(m_controller, m_guiRenderer3D, from, to); // Gtk::manage cannot be used for Gtk::Window
					m_guiScreenshot->signal_hide().connect(sigc::mem_fun(*this, &guiMain::on_screenshot_delete_clicked));
					m_guiScreenshot->show();

				} else {
					uint frameID = m_frameManager->getCurrentFrameID();
					m_guiScreenshot = new guiScreenshot(m_controller, m_guiRenderer3D, frameID, frameID); // Gtk::manage cannot be used for Gtk::Window
					m_guiScreenshot->signal_hide().connect(sigc::mem_fun(*this, &guiMain::on_screenshot_delete_clicked));
					m_guiScreenshot->show();
				}
			}

		} else {
			Gtk::MessageDialog dialog(*this, "No profile available", Gtk::MESSAGE_ERROR);
			dialog.run();
		}
	}
}


void guiMain::on_screenshot_delete_clicked() {
	delete m_guiScreenshot;
	m_guiScreenshot = NULL;
}


void guiMain::resetGUIOnline() {
	m_guiSeekbar->resetSeekbar();
	m_guiChart->initDataset();
	m_guiChart->set_sensitive(false);

	m_guiRenderer2D->stopRendering();
	m_guiRenderer3D->stopRendering();
	m_guiRenderer2D->init();
	m_guiRenderer3D->init();
	m_guiRenderer2D->startRendering(true);
	m_guiRenderer3D->startRendering(true);
}

/**
 * Reset to initial state (no frames available)
 */
void guiMain::resetGUIOffline() {
	m_guiRenderer2D->stopRendering();
	m_guiRenderer3D->stopRendering();
	m_guiSeekbar->resetSeekbar();
	m_guiChart->initDataset();
	m_guiChart->set_sensitive(false);
}

/**
 * Update state (frames available)
 */
void guiMain::updateGUIOffline() {
	m_guiSeekbar->initSeekbar();
	m_guiRenderer2D->init();
	m_guiRenderer2D->startRendering(false);
	m_guiRenderer3D->init();
	m_guiRenderer3D->startRendering(false);
	m_guiTreeView->init();
	if(!m_ToggleToolButton_Chart_View.get_active()) {
		m_ToggleToolButton_Chart_View.set_active(true);
	}
	m_guiChart->updateDataset();
	m_guiChart->set_sensitive(true);
	if(m_guiSlipDetection != NULL) {
		m_guiSlipDetection->setModeOffline();
	}
	setCurrentFrame(0);
}


void guiMain::setCurrentFrame(int frameID) {

	if(showSensorView) {
		m_controller->getRenderer()->renderFrame(frameID);
		m_guiSeekbar->setSliderPosition(frameID);
	}

	if(showChartView) {
		m_guiChart->setMarkerPosition(frameID);
	}

	if(m_guiSlipDetection != NULL) {
		m_guiSlipDetection->setCurrentFrameOffline(frameID);
	}

	m_guiTreeView->updateCharacteristics();
}


void guiMain::updateDataset() {
	m_guiChart->updateDataset();
}


void guiMain::setCharacteristics(std::vector<std::vector<int> >c) {
	characteristics = c;
}


std::vector<std::vector<int> > guiMain::getCharacteristics() {
	return characteristics;
}


// Maintain divider ration when resizing
void guiMain::on_vpaned_size_allocate(Gtk::Allocation& allocation) {
	if(m_resized) {
		int dividerPos = static_cast<int>(m_VPaned_Views_Ratio * allocation.get_height() );
		m_VPaned_Views.set_position(dividerPos);
		m_resized = false;
	} else {
		m_VPaned_Views_Ratio = static_cast<double>(m_VPaned_Views.get_position()) / allocation.get_height();

		// Enforce min/max divider positions
		bool limiter = false;
		int margin_top = 250; // in pixel
		int margin_bottom = 120;
		if(m_VPaned_Views.get_position() < margin_top) {
			m_VPaned_Views_Ratio = static_cast<double>(margin_top) / allocation.get_height();
			limiter = true;
		}
		else if(m_VPaned_Views.get_position() > allocation.get_height()-margin_bottom) {
			m_VPaned_Views_Ratio = static_cast<double>(allocation.get_height()-margin_bottom) / allocation.get_height();
			limiter = true;
		}
		if(limiter) {
			int dividerPos = static_cast<int>(m_VPaned_Views_Ratio * allocation.get_height() );
			m_VPaned_Views.set_position(dividerPos);
		}
	}
}


void guiMain::on_vpaned_realize() {
	m_resized = true;
	int dividerPos = static_cast<int>(m_VPaned_Views_Ratio * m_VPaned_Views.get_height() );
	m_VPaned_Views.set_position(dividerPos);
}


void guiMain::on_resize_notify(GdkEventConfigure* event) {
	m_resized = true;
}
