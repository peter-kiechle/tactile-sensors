#ifndef FRAMEGRABBERDSA_HPP_
#define FRAMEGRABBERDSA_HPP_

#include <boost/thread.hpp>

#include "sdh/dsa.h"
#include "sdh/util.h"

USING_NAMESPACE_SDH

class FrameManager;

/**
 * @class FrameGrabberDSA
 * @brief A frame grabber/recorder class for tactile sensor based on boost thread
 */
class FrameGrabberDSA {

private:

	cDSA* ts;
	FrameManager *frameManager;

	double framerate;
	bool request_single_frames;
	double frame_interval;
	double remaining_time;
	int nb_errors;
	int nb_frames;

	boost::thread grabber_thread;

	bool paused;
	boost::mutex paused_mutex;
	boost::condition_variable paused_changed;

	bool recording;
	boost::mutex recording_mutex;

	cSimpleTime start_time;
	cSimpleTime current_time;
	cSimpleTime last_time;
	cSimpleTime last_time_temperature;

	void conditionalBreakpoint();

public:

	/**
	 * Constructor (not recording yet).
	 * @param dsa The tactile sensor controller.
	 */
	FrameGrabberDSA(cDSA *dsa);


	/**
	 * Constructor (without frame manager).
	 * @param dsa The tactile sensor controller.
	 * @param fm The frame manager.
	 */
	FrameGrabberDSA(cDSA* dsa, FrameManager *fm);


	/**
	 * Deconstructor: Interrupts thread in as soon as possible.
	 */
	~FrameGrabberDSA();


	/**
	 * Sets the frame manager.
	 * @param fm The frame manager.
	 * @return void
	 */
	void setFrameManager(FrameManager *fm);


	/**
	 * Sets the frame rate.
	 * @details For frame rates < 30, tactile sensor frames are manually requested (pull mode).
	 *          Otherwise, the DSA controller switches to an automatic push mode.
	 * @param frameRate The desired frame rate.
	 * @return void
	 */
	void setFramerate(double frameRate);


	/**
	 * Initializes DSA Controller and starts the execution of the grabber thread.
	 * @details For frame rates < 30, tactile sensor frames are manually requested (pull mode).
	 *          Otherwise, the DSA controller switches to an automatic push mode.
	 * @param frameRate The desired frame rate.
	 * @param startPaused Should the thread start paused or immediately start grabbing?
	 * @param startRecording Should the grabbing thread immediately start recording?
	 * @return void
	 */
	void start(double frameRate, bool startPaused = false, bool startRecording = false);


	/**
	 * Executes the grabbing / recording thread.
	 * @return void
	 */
	void execute();


	/**
	 * Stop DSA push-mode (if active) and halt thread execution.
	 * @return void
	 */
	void pause();


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
