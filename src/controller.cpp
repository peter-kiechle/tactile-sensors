#include <boost/program_options.hpp>

#include "controller.h"
#include "guiMain.h"
#include "guiRenderer.h"

Controller::Controller(int argc, char* argv[]) {
	setlocale(LC_ALL, "C"); // Standard C locale

	frameManager = new FrameManager();
	ts = NULL;
	hand = NULL;
	frameGrabberDSA = NULL;
	frameGrabberSDH = NULL;

	int mydebug_level = 0;

	// Hardcoded serial numbers of joint- and sensor controllers of connected SCHUNK hand(s)
	SDH_Device = new Device("Left SDH-2", SDH2, 48);
	//SDH_Device = Device("Right SDH-2", SDH2, 47);
	DSA_Device = new Device("Left DSACON32-M", DSACON32m, 496697431);
	//DSA_Device = Device("Right DSACON32-M", DSACON32m, 109576763);

	list<Device*> deviceList;
	deviceList.push_back(SDH_Device);
	deviceList.push_back(DSA_Device);

	// Try to identify devices automatically
	libExtension = new Ext(mydebug_level, deviceList);
	libExtension->IdentifyDevices();

	std::string filename;
	namespace po = boost::program_options;
	try
	{
		po::options_description description("Usage");
		description.add_options() // Note: The order of options is implicitly defined
	    								  ("help,h", "Display this information")
	    								  ("input-file,i", po::value<std::string>(), "Input pressure profile")
	    								  ;

		// Missing option name is mapped to "input-file"
		po::positional_options_description pos;
		pos.add("input-file", -1); // number of consecutive values to be associated with the option

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(description).positional(pos).run(), vm);

		if(vm.count("help")) { // Print help before po::notify() complains about missing required variables
			std::cout << description << "\n";
		}

		po::notify(vm); // Assign the specified variables, throws an error if required variables are missing

		if(vm.count("input-file")) {
			filename = vm["input-file"].as<std::string>();
			std::cout << "filename: " << filename << std::endl;
		}
	}
	catch(std::exception& e) {
		std::cerr << "Error: " << e.what() << "\n";
	}
	catch(...) {
		std::cerr << "Unhandled Exception in main!" << "\n";
	}

	// Load specified pressure profile
	loadProfile(filename);

	// handle command line options: set defaults first, then overwrite by parsing actual command line
	options = new cSDHOptions( "general sdhcom_serial sdhcom_common sdhcom_esdcan sdhcom_peakcan sdhcom_cancommon sdhcom_tcp dsacom" );
	options->debug_level = mydebug_level;
	options->Parse(argc, argv, "not available", "test", "version", cSDH::GetLibraryName(), cSDH::GetLibraryRelease() );

	// Overwrite default device options
	if(!SDH_Device->device_format_string.empty()) {
		strncpy(options->sdh_rs_device, SDH_Device->device_format_string.c_str(), options->MAX_DEV_LENGTH);
	} else {
		options->sdh_rs_device[0] = '\0';
	}
	if(!DSA_Device->device_format_string.empty()) {
		strncpy(options->dsa_rs_device, DSA_Device->device_format_string.c_str(), options->MAX_DEV_LENGTH);
	} else {
		options->dsa_rs_device[0] = '\0';
	}

	// Initialize debug message printing:
	cdbg = new cDBG( options->debug_level > 0, "red", options->debuglog );
	g_sdh_debug_log = options->debuglog;
	*cdbg << "Debug messages of " << argv[0] << " are printed like this.\n";


	Glib::thread_init();
	Gtk::Main kit(argc, argv);	// GTK initialization
	Gtk::GL::init(argc, argv); // gtkglextmm
	guiMain gui(this);
	Gtk::Main::run(gui);
}

Controller::~Controller() {

	disconnectSDH();
	disconnectDSA();

	delete SDH_Device;
	delete DSA_Device;
	delete cdbg;
	delete options;
	delete libExtension;
}


bool Controller::isAvailableSDH() {
	if(options->sdh_rs_device[0] == '\0' ) {
		return false;
	} else {
		return true;
	}
}

bool Controller::isAvailableDSA() {
	if(options->dsa_rs_device[0] == '\0' ) {
		return false;
	} else {
		return true;
	}
}


void Controller::connectSDH() {

	if(!isConnectedSDH()) {
		try {
			// Create an instance of the joint controller
			fprintf(stderr, "Connecting to SDH-2 joint controller...");

			hand = new cSDH( false, false, options->debug_level );
			*cdbg << "Successfully created cSDH instance\n";

			// Open configured communication to the SDH device
			options->OpenCommunication(*hand);

			fprintf(stderr, "OK\n");
			*cdbg << "Successfully opened communication to SDH\n";

			frameGrabberSDH = new FrameGrabberSDH(hand, frameManager);
			frameManager->setSDH(hand);
			frameManager->setFrameGrabberSDH(frameGrabberSDH);


		} catch (cSDHLibraryException* e) {
			cerr << "\nCaught exception from SDHLibrary: " << e->what() << ". Giving up!\n";
			delete e;
		} catch (...) {
			cerr << "\nCaught unknown exception, giving up\n";
		}
	}
}


void Controller::connectDSA() {

	if(!isConnectedDSA()) {
		cerr << "dsa_rs_device: "<< options->dsa_rs_device << endl;
		try {
			// Create instance of tactile sensor
			printf("Connecting to SDH-2 tactile sensor controller. This may take up to 8 seconds...");
			ts = new cDSA(options->debug_level, options->dsaport, options->dsa_rs_device );

			printf("OK\n");
			*cdbg << "Successfully created cDSA instance\n";
		} catch (cSDHLibraryException* e) {
			cerr << "\nCaught exception from SDHLibrary: " << e->what() << ". Giving up!\n";
			delete e;
		} catch (...) {
			cerr << "\ncaught unknown exception, giving up\n";
		}

		printf("DSA is now connected!\n");

		std::cout << "Sensor Info:\n";
		std::cout << ts->GetSensorInfo() << std::endl;
		std::cout << "Matrix Info:\n";
		uint numMatrices = ts->GetSensorInfo().nb_matrices;
		for(uint m = 0; m < numMatrices; m++) {
			printf("Matrix: %d\n", m);
			std::cout << ts->GetMatrixInfo(m) << std::endl;
		}

		/*
		// Set all cells inactive
		for(uint m = 0; m < getNumMatrices(); m++) {
			for(uint y = 0; y < matrices[m].cells_y; y++) {
				for(uint x = 0; x < matrices[m].cells_x; x++) {
					ts->SetMatrixMaskCell(m, x, y, false);
				}
			}
		}

		// Activate matrix 5
		//	uint m = 5;
		//	for(uint y = 0; y < matrices[m].cells_y; y++) {
		//		for(uint x = 0; x < matrices[m].cells_x; x++) {
		//			ts->SetMatrixMaskCell(m, x, y, true);
		//		}
		//	}

		ts->SetMatrixMaskCell(3, 2, 9, true);
		ts->SetMatrixMaskCell(3, 3, 9, true);

		printf("Setting Matrix Masks!!!!\n");
		ts->SetMatrixMasks();

		printf("\n\nDyamic Matrix Masks:\n");
		for(uint m = 0; m < numMatrices; m++) {
			std::cout << GetMatrixMaskString(MASK_DYNAMIC, m, true) << std::endl;
		}
		 */

		// Query matrix sensitivity + threshold
		for ( uint m = 0; m < numMatrices; m++ ) {
			printf("\n\nMatrix %d:\n", m);
			cDSA::sSensitivityInfo sensitivity_info = ts->GetMatrixSensitivity(m);
			cout << "  sensitivity         = " << sensitivity_info.cur_sens  << "\n";
			cout << "  factory_sensitivity = " << sensitivity_info.fact_sens << "\n";
			cout << "  threshold           = " << ts->GetMatrixThreshold(m)   << "\n";
		}

		frameGrabberDSA = new FrameGrabberDSA(ts, frameManager);
		frameManager->setDSA(ts);
		frameManager->setFrameGrabberDSA(frameGrabberDSA);
	}
}


void Controller::disconnectSDH() {
	if(frameGrabberSDH) {
		frameGrabberSDH->finish();
		delete frameGrabberSDH;
		frameGrabberSDH = NULL;
	}
	if(hand) {
		hand->Close();
		*cdbg << "Successfully closed connection to joint controller\n";
		delete hand;
		hand = NULL;
	}
}


void Controller::disconnectDSA() {
	if(frameGrabberDSA) {
		frameGrabberDSA->finish();
		delete frameGrabberDSA;
		frameGrabberDSA = NULL;
	}
	if(ts) {
		ts->Close();
		*cdbg << "Successfully closed connection to tactile sensor controller\n";
		delete ts;
		ts = NULL;
	}
}


bool Controller::isConnectedSDH() {
	if(hand) {
		return hand->IsOpen();
	} else {
		return false;
	}
}

bool Controller::isConnectedDSA() {
	if(ts) {
		return ts->IsOpen();
	} else {
		return false;
	}
}


void Controller::loadProfile(const std::string& filename) {
	if(!filename.empty()) {
		// Check if file extension is dsa
		m_profilePath = boost::filesystem::path(filename);
		std::string extension = m_profilePath.extension().string();
		if(extension == ".dsa") {
			getFrameManager()->loadFrames(filename);
		}
	}
}


boost::filesystem::path Controller::getProfilePath() {
	return m_profilePath;
}


std::string Controller::getProfilePathName() {
	return m_profilePath.string();
}


std::string Controller::getProfileDirectory() {
	return m_profilePath.parent_path().string();
}

std::string Controller::getProfileName() { // With extension
	return m_profilePath.filename().string();
}


std::string Controller::getProfileBaseName() { // Without extension
	return m_profilePath.stem().string();
}


std::string Controller::getProfileExtension() {
	return m_profilePath.extension().string();
}


void Controller::setRenderer(guiRenderer *r) {
	renderer = r;
}


vector<double> Controller::getPreshape(int graspID, double closeRatio) {
	vector<double> hand_pose;
	switch(graspID) {

	case 0: // Parallel (parallel fingertips)
		hand_pose.push_back(0); // Rotational axis (Finger 0 + 2)
		hand_pose.push_back(-75+closeRatio*82); // Finger 0
		hand_pose.push_back(75-closeRatio*82);
		hand_pose.push_back(-75+closeRatio*82); // Finger 1
		hand_pose.push_back(75-closeRatio*82);
		hand_pose.push_back(-75+closeRatio*82); // Finger 2
		hand_pose.push_back(75-closeRatio*82);
		return hand_pose;

	case 1: // Cylindrical (tilted fingertips)
		hand_pose.push_back(0); // Rotational axis (Finger 0 + 2)
		hand_pose.push_back(-30+closeRatio*30); // Finger 0
		hand_pose.push_back(30+closeRatio*35);
		hand_pose.push_back(-30+closeRatio*30); // Finger 1
		hand_pose.push_back(30+closeRatio*35);
		hand_pose.push_back(-30+closeRatio*30); // Finger 2
		hand_pose.push_back(30+closeRatio*35);
		return hand_pose;

	case 2: // Centrical (parallel fingertips)
		hand_pose.push_back(60); // Rotational axis (Finger 0 + 2)
		hand_pose.push_back(-75+closeRatio*82); // Finger 0
		hand_pose.push_back(75-closeRatio*82);
		hand_pose.push_back(-75+closeRatio*82); // Finger 1
		hand_pose.push_back(75-closeRatio*82);
		hand_pose.push_back(-75+closeRatio*82); // Finger 2
		hand_pose.push_back(75-closeRatio*82);
		return hand_pose;

	case 3: // Spherical (tilted fingertips)
		hand_pose.push_back(60); // Rotational axis (Finger 0 + 2)
		hand_pose.push_back(-40+closeRatio*25); // Finger 0
		hand_pose.push_back(40+closeRatio*15);
		hand_pose.push_back(-40+closeRatio*25); // Finger 1
		hand_pose.push_back(40+closeRatio*15);
		hand_pose.push_back(-40+closeRatio*25); // Finger 2
		hand_pose.push_back(40+closeRatio*15);
		return hand_pose;

	case 4: // Pinch grip (parallel fingertips)
		hand_pose.push_back(90); // Rotational axis (Finger 0 + 2)
		hand_pose.push_back(-72 + closeRatio * 83); // Finger 0
		hand_pose.push_back(72 - closeRatio * 83);
		hand_pose.push_back(-90); // Finger 1
		hand_pose.push_back(0);
		hand_pose.push_back(-72 + closeRatio * 83); // Finger 2
		hand_pose.push_back(72 - closeRatio * 83);
		return hand_pose;

	case 5: // Pinch grip (tilted fingertips)
		hand_pose.push_back(90); // Rotational axis (Finger 0 + 2)
		hand_pose.push_back(-40+closeRatio*40); // Finger 0
		hand_pose.push_back(10+closeRatio*13);
		hand_pose.push_back(-90); // Finger 1
		hand_pose.push_back(0);
		hand_pose.push_back(-40+closeRatio*40); // Finger 2
		hand_pose.push_back(10+closeRatio*13);
		return hand_pose;

	case 6: // Awkward position for measurements (pressure calibration)
		hand_pose.push_back(90.0); // Rotational axis (Finger 0 + 2)
		hand_pose.push_back(0); // Finger 0
		hand_pose.push_back(0);
		hand_pose.push_back(-90); // Finger 1
		hand_pose.push_back(-90);
		hand_pose.push_back(-90); // Finger 2
		hand_pose.push_back(-90);

		return hand_pose;

	default:
		hand_pose.push_back(0);
		hand_pose.push_back(0);
		hand_pose.push_back(0);
		hand_pose.push_back(0);
		hand_pose.push_back(0);
		hand_pose.push_back(0);
		hand_pose.push_back(0);
		return hand_pose;
	}
}


void Controller::grasp(int graspID, double closeRatio, double velocity) {

	// Pause grabbing for a moment to prevent conflicts on the serial port while sending grasp command
	bool isGrabbing = false;

	if(frameGrabberSDH->isCapturing()) {
		isGrabbing = true;
		frameGrabberSDH->pauseBlocking();
	}

	// Switch to "pose" controller mode and set default velocities first:
	hand->SetController(hand->eCT_POSE);
	hand->SetAxisTargetVelocity(hand->All, velocity);
	hand->SetVelocityProfile( cSDH::eVP_INVALID );

	vector<double> hand_pose = getPreshape(graspID, closeRatio);

	hand->SetAxisTargetAngle(hand->all_real_axes, hand_pose);
	hand->MoveAxis(hand->all_real_axes, false); // Move axes there non sequentially:
	// The last call returned immediately so we now have time to
	// do something else while the hand is moving:
	// ... insert any calculation here ...

	// Resume grabbing
	if(isGrabbing) {
		frameGrabberSDH->resume();
	}

	bool finished_movement = false;
	cSimpleTime current_time;
	cSimpleTime last_time_axis_state;

	// Continuously check if end position was reached
	while(true) {

		current_time.StoreNow();

		if(frameGrabberSDH->isCapturing()) {
			isGrabbing = true;
			frameGrabberSDH->pauseBlocking();
		}

		// Check for busy states of position- and velocity controller modes
		if(last_time_axis_state.Elapsed(current_time) >= 0.1) {
			std::vector<cSDH::eAxisState> states = hand->GetAxisActualState( hand->all_real_axes );
			std::vector<cSDH::eAxisState>::const_iterator it;
			finished_movement = true;
			for(it = states.begin(); it != states.end(); it++) {
				bool busy = *it == cSDH::eAS_POSITIONING || *it == cSDH::eAS_SPEED_MODE;
				finished_movement = finished_movement && !busy;
			}
		}

		// End position reached
		if(finished_movement) {
			fprintf(stderr, "End position reached\n");
			hand->Stop();
			break;
		}

		// Resume grabbing
		if(isGrabbing) {
			frameGrabberSDH->resume();
		}
	}

	// Resume grabbing
	if(isGrabbing) {
		frameGrabberSDH->resume();
	}
}


boost::tuple<bool, float> Controller::graspReactive(int graspID, double velocity, double limit) {

	// Pause grabbing for a moment to prevent conflicts on the serial port while sending grasp command
	bool isGrabbingSDH = false;

	if(frameGrabberSDH->isCapturing()) {
		isGrabbingSDH = true;
		frameGrabberSDH->pauseBlocking();
	}

	// Switch to "pose" controller mode and set default velocities first:
	hand->SetController(hand->eCT_POSE);
	hand->SetAxisTargetVelocity(hand->All, velocity);
	hand->SetVelocityProfile( cSDH::eVP_RAMP );

	// Get predefined target joint angles
	vector<double> hand_pose = getPreshape(graspID, 1.0);

	hand->SetAxisTargetAngle(hand->all_real_axes, hand_pose);
	hand->MoveAxis(hand->all_real_axes, false); // Move axes there non sequentially:
	// The last call returned immediately so we now have time to
	// do something else while the hand is moving:
	// ... insert any calculation here ...


	// Resume grabbing
	if(isGrabbingSDH) {
		frameGrabberSDH->resume();
	}

	bool finished_movement = false;
	cSimpleTime current_time;
	cSimpleTime last_time_axis_state;

	// Continuously check for break conditions while the hand is closing:
	// Tactile sensor limit reached?
	// End position reached?
	float max = 0.0;
	while(true) {

		current_time.StoreNow();

		if(frameGrabberSDH->isCapturing()) {
			isGrabbingSDH = true;
			frameGrabberSDH->pauseBlocking();
		}

		// Check for busy states of position- and velocity controller modes
		// i.e. is hand still moving or at least trying to
		if(last_time_axis_state.Elapsed(current_time) >= 0.1) {
			std::vector<cSDH::eAxisState> states = hand->GetAxisActualState( hand->all_real_axes );
			std::vector<cSDH::eAxisState>::const_iterator it;
			finished_movement = true;
			for(it = states.begin(); it != states.end(); it++) {
				bool busy = *it == cSDH::eAS_POSITIONING || *it == cSDH::eAS_SPEED_MODE;
				finished_movement = finished_movement && !busy;
			}
		}
		// End position reached
		if(finished_movement) {
			fprintf(stderr, "End position reached\n");
			hand->Stop();
			break;
		}

		// Get max value of current frame
		TSFrame* currentFilteredFrame = frameManager->getCurrentFilteredFrame();
		for(uint i = 0; i < frameManager->getNumCells(); i++) {
			if(currentFilteredFrame->cells[i] > max) {
				max = currentFilteredFrame->cells[i];
			}
		}

		// Limit reached
		if(max > limit) {
			fprintf(stderr, "Limit reached: %f\n", max);
			hand->Stop();
			break;
		}

		// Resume grabbing
		if(isGrabbingSDH) {
			frameGrabberSDH->resume();
		}

	}
	// Resume grabbing
	if(isGrabbingSDH) {
		frameGrabberSDH->resume();
	}
	return boost::make_tuple(finished_movement, max);
}


cDSA* Controller::getDSA() {
	return ts;
}


cSDH* Controller::getSDH() {
	return hand;
}


FrameManager* Controller::getFrameManager() {
	return frameManager;
}


FrameGrabberDSA* Controller::getFrameGrabberDSA() {
	return frameGrabberDSA;
}


FrameGrabberSDH* Controller::getFrameGrabberSDH() {
	return frameGrabberSDH;
}


guiRenderer* Controller::getRenderer() {
	return renderer;
}
