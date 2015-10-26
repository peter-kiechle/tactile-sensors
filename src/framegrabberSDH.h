#ifndef FRAMEGRABBERSDH_HPP_
#define FRAMEGRABBERSDH_HPP_

#include <boost/thread.hpp>

#include "sdh/sdh.h"
#include "sdh/simpletime.h"
#include "sdh/util.h"

USING_NAMESPACE_SDH

class FrameManager;
class Controller;

/**
 * @class FrameGrabberSDH
 * @brief A frame grabber/recorder class for joint angles and temperatures based on boost thread
 */
class FrameGrabberSDH {

private:

	cSDH* hand;
	FrameManager *frameManager; // frameManager is optional ("memoryless" mode)
	Controller *controller;

	double framerateJointAngles;
	double framerateTemperature;
	double frameIntervalJointAngles;
	double frameIntervalTemperature;

	int nb_errors;
	int nb_frames;

	boost::thread grabber_thread;

	bool paused;
	boost::mutex paused_mutex;
	boost::condition_variable paused_changed;

	bool line_clear;
	boost::mutex mutex_line_clear;
	boost::condition_variable line_clear_changed;

	bool recording;
	boost::mutex recording_mutex;

	bool enableTemperature;
	bool enableJointAngle;

	cSimpleTime current_time;
	cSimpleTime last_time_temperature;
	cSimpleTime last_time_joint_angles;

	void conditionalBreakpoint();

public:

	/**
	 * Constructor (without frame manager).
	 * @param sdh The SCHUNK SDH-2.
	 */
	FrameGrabberSDH(cSDH* sdh);


	/**
	 * Constructor (ready to record).
	 * @param dsa The tactile sensor controller.
	 * @param fm The frame manager.
	 */
	FrameGrabberSDH(cSDH* sdh, FrameManager *fm);


	/**
	 * Deconstructor: Interrupts thread in as soon as possible.
	 */
	~FrameGrabberSDH();

	/**
	 * Sets the frame manager.
	 * @param fm The frame manager.
	 * @return void
	 */
	void setFrameManager(FrameManager *fm);

	/**
	 * Sets the frame rate of the joint angle requests.
	 * @param frameRate The desired frame rate.
	 */
	void setFramerateJointAngles(double frameRate);


	/**
	 * Sets the frame rate of the joint angle requests.
	 * @param frameRate The desired frame rate.
	 */
	void setFramerateTemperature(double frameRate);


	/**
	 * Separate flag to capture/record temperature readings
	 * @param enable The capturing state.
	 */
	void setTemperature(bool enable);


	/**
	 * Separate flag to capture/record joint angle readings
	 * @param enable The capturing state.
	 */
	void setJointAngle(bool enable);


	/**
	 * Initializes the SDH-2 and starts the execution of the grabber thread.
	 * @param FPSJointAngles The desired joint angle frame rate.
	 * @param FPSTemperatureThe desired temperature frame rate.
	 * @param startPaused Should the thread start paused or immediately start grabbing?
	 * @param startRecording Should the grabbing thread immediately start recording?
	 */
	void start(double FPSJointAngles, double FPSTemperature, bool startPaused = false, bool startRecording = false);


	/**
	 * Executes the grabbing / recording thread.
	 * @return void
	 */
	void execute();


	/**
	 * Halt thread execution.
	 * @return void
	 */
	void pause();


	/**
	 * Halt thread execution and wait for end of transmission.
	 * @return void
	 */
	void pauseBlocking();


	/**
	 * Resume thread execution with current configuration.
	 * @return void
	 */
	void resume();


	/**
	 * Stops the grabber thread.
	 * Interrupts thread during next wait/sleep state.
	 * @return void
	 */
	void finish();


	/**
	 * Automatically request new frames and store them permanently.
	 * @return void
	 */
	void enableRecording();


	/**
	 * Pause storing of new frames.
	 * @return void
	 */
	void suspendRecording();


	/**
	 * Reports if the thread is running or paused.
	 * @return The state.
	 */
	bool isRunning();


	/**
	 * Reports if the thread is capturing or not.
	 * @return The state.
	 */
	bool isCapturing();


	/**
	 * Reports if the thread is recording or not.
	 * @return The state.
	 */
	bool isRecording();


	/**
	 * Returns the number of already captured frames.
	 * @return The number of already captured frames.
	 */
	int getFrameNumber() {return nb_frames;}

};

#endif /* FRAMEGRABBER_HPP_ */
