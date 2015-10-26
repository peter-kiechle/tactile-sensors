#ifndef CONTROLLER_H_
#define CONTROLLER_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <locale.h>
#include <gtkglmm.h>

#include <boost/filesystem.hpp>

#include "sdh/sdh.h"
#include "sdh/dsa.h"
#include "sdh/util.h"
#include "sdhoptions.h"

#include "extension.h"
#include "framemanager.h"
#include "framegrabberDSA.h"
#include "framegrabberSDH.h"
#include "calibration.h"

USING_NAMESPACE_SDH
using namespace std;

class guiRenderer;

/**
 * @class Controller
 * @brief Implements the controller of the GUI's model–view–controller (MVC) pattern
 * @note In case the GUI is not of interest for your project, this class may still serve as an example application.
 */
class Controller {

private:

	// Device identifiers
	Device *SDH_Device;
	Device *DSA_Device;

	// Device instances
	cSDH *hand;
	cDSA *ts;

	Ext *libExtension;

	Calibration calibration;

	cSDHOptions *options;
	cDBG *cdbg;

	FrameGrabberDSA *frameGrabberDSA;
	FrameGrabberSDH *frameGrabberSDH;
	FrameManager *frameManager;
	guiRenderer *renderer;
	boost::filesystem::path m_profilePath;

public:

	/**
	 * Constructor automatically initializes a whole lot of different subsystems
	 * @note You will have to change the hard-coded serial numbers according to your SDH-2
	 */
	Controller(int argc, char* argv[]);
	~Controller();


	/**
	 * Reports if the SDH-2 was found.
	 * @return The state.
	 */
	bool isAvailableSDH();


	/**
	 * Reports if a the DSA controller was found.
	 * @return The state.
	 */
	bool isAvailableDSA();


	/**
	 * Creates an instance of the SHD-2 and tries to open the communication.
	 * @return void
	 */
	void connectSDH();


	/**
	 * Creates an instance of the DSA controller and tries to open the communication.
	 * @return void
	 */
	void connectDSA();


	/**
	 * Disconnects the SDH-2 and deletes the instance.
	 * @return void
	 */
	void disconnectSDH();


	/**
	 * Disconnects the DSA controller and deletes the instance.
	 * @return void
	 */
	void disconnectDSA();


	/**
	 * Reports if a the SDH-2 is connected and ready to go.
	 * @return The state.
	 */
	bool isConnectedSDH();


	/**
	 * Reports if a the DSA controller is connected and ready to go.
	 * @return The state.
	 */
	bool isConnectedDSA();


	/**
	 * Opens the specified *.dsa file and loads it's contents.
	 * @param filename The filename of the *.dsa profile
	 * @return void
	 */
	void loadProfile(const std::string& filename);


	/// Getter/Setter
	cDSA* getDSA();
	cSDH* getSDH();
	FrameManager* getFrameManager();
	FrameGrabberDSA* getFrameGrabberDSA();
	FrameGrabberSDH* getFrameGrabberSDH();

	guiRenderer* getRenderer();
	void setRenderer(guiRenderer *r);

	boost::filesystem::path getProfilePath();
	std::string getProfilePathName();
	std::string getProfileDirectory();
	std::string getProfileName(); // With extension
	std::string getProfileBaseName(); // Without extension
	std::string getProfileExtension();

	Calibration& getCalibration() { return calibration; }


	/**
	 * Gets joint angles of preshaped grasp
	 * @param graspID The graspID, see source for description.
	 * @param closeRatio Open/Close ratio in the range [0.0, 1.0].
	 * @return void
	 */
	vector<double> getPreshape(int graspID, double closeRatio);


	/**
	 * Reimplementation of the SDHLibrary's grasping routine.
	 * @details This function is non-blocking and continuously queries the axis states itself.
	 *          Thus, joint angles can be recorded while grasping.
	 *          In contrast, the original function executes the grasp and only returns when the grasp is completed.
	 * @param graspID The graspID, see source for description.
	 * @param closeRatio Open/Close ratio in the range [0.0, 1.0].
	 * @param velocity The grasping speed.
	 * @return void
	 */
	void grasp(int graspID, double closeRatio, double velocity);


	/**
	 * Reactive grasping routine similar to grasp() that also takes the tactile sensors into account.
	 * @details This function is non-blocking and continuously queries the axis states itself.
	 *          Thus, joint angles can be recorded while grasping.
	 *          In contrast, the original function executes the grasp and only returns when the grasp is completed.
	 * @param graspID The graspID, see source for description.
	 * @param closeRatio Open/Close ratio in the range [0.0, 1.0].
	 * @param velocity The grasping speed.
	 * @param velocity Maximum sensor value limit.
	 * @return Tuple of the state of the grasp and maximum sensor value.
	 */
	boost::tuple<bool, float> graspReactive(int graspID, double velocity, double limit);
};

#endif /* CONTROLLER_H_ */
