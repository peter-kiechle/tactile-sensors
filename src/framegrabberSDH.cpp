#include <boost/thread.hpp>
#include <boost/chrono.hpp>

#include "framegrabberSDH.h"
#include "framemanager.h"


FrameGrabberSDH::FrameGrabberSDH(cSDH* sdh) {
	hand = sdh;
	frameManager = NULL;
	recording = false;
	paused = true;
	line_clear = true;
	enableTemperature = false;
	enableJointAngle = false;
}


FrameGrabberSDH::FrameGrabberSDH(cSDH* sdh, FrameManager *fm) {
	hand = sdh;
	frameManager = fm;
	recording = false;
	paused = true;
	enableTemperature = false;
	enableJointAngle = false;
}


FrameGrabberSDH::~FrameGrabberSDH() {}


void FrameGrabberSDH::setFrameManager(FrameManager *fm) {
	frameManager = fm;
}


void FrameGrabberSDH::setFramerateJointAngles(double frameRate) {
	framerateJointAngles = frameRate;
	frameIntervalJointAngles = 1.0 / framerateJointAngles;
}


void FrameGrabberSDH::setFramerateTemperature(double frameRate) {
	framerateTemperature = frameRate;
	frameIntervalTemperature = 1.0 / framerateTemperature;
}


void FrameGrabberSDH::setTemperature(bool enable) {
	enableTemperature = enable;
}


void FrameGrabberSDH::setJointAngle(bool enable) {
	enableJointAngle = enable;
}


void FrameGrabberSDH::start(double FPSJointAngles, double FPSTemperature, bool startPaused, bool startRecording) {

	setFramerateJointAngles(FPSJointAngles);
	setFramerateTemperature(FPSTemperature);

	paused = startPaused;

	if(startRecording) {
		enableRecording();
	}

	current_time.StoreNow();
	last_time_temperature = current_time;

	grabber_thread = boost::thread(boost::bind(&FrameGrabberSDH::execute, this));
}


void FrameGrabberSDH::execute() {

	printf("Entering SDH thread execution\n");

	nb_frames = 0;
	nb_errors = 0;
	while(true) {
		nb_frames++;
		try	{

			conditionalBreakpoint(); // check preconditions for next iteration

			current_time.StoreNow();

			// Request temperature readings from sdh
			if(enableTemperature) {
				if(last_time_temperature.Elapsed(current_time) >= frameIntervalTemperature) {
					frameManager->requestTemperatureFrame(recording);
					//std::cout << "    Temperature framerate=" << std::setprecision(2) << std::fixed << (1.0/last_time_temperature.Elapsed(current_time)) << "Hz nb_frames=" << nb_frames << " nb_errors=" << nb_errors << " (" << ((100.0*nb_errors)/nb_frames) << "%)\n";
					//std::cout.flush();
					last_time_temperature = current_time;
				}
			}

			// Request joint angle readings
			if(enableJointAngle) {
				if(last_time_joint_angles.Elapsed(current_time) >= frameIntervalJointAngles) {
					frameManager->requestJointAngleFrame(recording);
					//std::cout << "    Joint Angle framerate=" << std::setprecision(2) << std::fixed << (1.0/last_time_joint_angles.Elapsed(current_time)) << "Hz nb_frames=" << nb_frames << " nb_errors=" << nb_errors << " (" << ((100.0*nb_errors)/nb_frames) << "%)\n";
					//std::cout.flush();
					last_time_joint_angles = current_time;
				}
			}

		} catch (cDSAException* e) {
			nb_errors++;
			std::cerr << "Caught and ignored cDSAException: " << e->what() << " nb_errors=" << nb_errors << "  Frame " << nb_frames << "\n";
			delete e;
		}
	}
	printf("Exiting SDH thread\n");
}


void FrameGrabberSDH::conditionalBreakpoint() {

	// There should be no active transmissions here anymore
	boost::unique_lock<boost::mutex> line_clear_lock(mutex_line_clear);
	line_clear = true;
	line_clear_changed.notify_one();
	line_clear_lock.unlock();

	// Grabber thread blocks while paused
	boost::unique_lock<boost::mutex> pause_lock(paused_mutex);
	while(paused) { // Loop to catch spurious wake-ups
		paused_changed.wait(pause_lock); // wait() unlocks mutex automatically
	}

	// Transmissions may begin again
	line_clear_lock.lock();
	line_clear = false;
	line_clear_changed.notify_one();
	line_clear_lock.unlock();
}


void FrameGrabberSDH::pause() {

	// Change pause state as well as condition variable
	boost::unique_lock<boost::mutex> lock(paused_mutex);
	paused = true;
	paused_changed.notify_one();
	lock.unlock(); // Should be unnecessary as lock goes out of scope

}


void FrameGrabberSDH::pauseBlocking() {

	pause();

	// Wait until all transmissions on the serial port have stopped (or timeout)
	boost::unique_lock<boost::mutex> line_clear_lock(mutex_line_clear);
	while(paused && !line_clear) { // Loop to catch spurious wake-ups
		line_clear_changed.wait(line_clear_lock); // wait() unlocks mutex automatically
	}
}


void FrameGrabberSDH::resume() {
	assert(paused == true && "Frame grabber has to be paused first to be resumed");

	// Change pause state as well as condition variable
	boost::unique_lock<boost::mutex> lock(paused_mutex);
	paused = false;
	paused_changed.notify_one();
	lock.unlock(); // Should be unnecessary as lock goes out of scope
}


void FrameGrabberSDH::finish() {
	pause();
	grabber_thread.interrupt(); // Ask thread to stop
	grabber_thread.join(); // Wait for it
}


void FrameGrabberSDH::enableRecording() {
	assert(frameManager != NULL && "Unable to start recording without a FrameManager instance");
	boost::unique_lock<boost::mutex> lock(recording_mutex);
	recording = true;
	printf("SDH Recording enabled\n");
}


void FrameGrabberSDH::suspendRecording() {
	boost::unique_lock<boost::mutex> lock(recording_mutex);
	recording = false;
	printf("SDH Recording suspended\n");
}


bool FrameGrabberSDH::isRunning() {
	return grabber_thread.try_join_for(boost::chrono::milliseconds(0)); // *ugly hack*
}

bool FrameGrabberSDH::isCapturing() {
	return !paused;
}

bool FrameGrabberSDH::isRecording() {
	return recording;
}
