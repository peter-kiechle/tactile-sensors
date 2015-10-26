#include <boost/chrono.hpp>

#include "framegrabberDSA.h"
#include "framemanager.h"

FrameGrabberDSA::FrameGrabberDSA(cDSA* dsa) {
	ts = dsa;
	frameManager = NULL;
	recording = false;
	paused = true;
}

FrameGrabberDSA::FrameGrabberDSA(cDSA* dsa, FrameManager *fm) {
	ts = dsa;
	frameManager = fm;
	recording = false;
	paused = true;
}

FrameGrabberDSA::~FrameGrabberDSA() {}


void FrameGrabberDSA::setFrameManager(FrameManager *fm) {
	frameManager = fm;
}


void FrameGrabberDSA::setFramerate(double frameRate) {
	framerate = frameRate;
	request_single_frames = framerate < 30.0;

	if(request_single_frames) {
		// Make remote tactile sensor controller stop sending data automatically as fast as possible (prepare for DSA pull-mode):
		ts->SetFramerate(0, true, false); // with RLE
	} else {
		// Make remote tactile sensor controller send data automatically as fast as possible (DSA push-mode):
		ts->SetFramerate(1, true, true); // with RLE
	}
}


void FrameGrabberDSA::start(double frameRate, bool startPaused, bool startRecording) {
	framerate = frameRate;
	frame_interval = 1.0 / framerate;
	paused = startPaused;

	// Initialize DSA Controller
	if(startPaused) {
		ts->SetFramerateRetries(0, true, false, 3, true); // Disable DSA push-mode
	} else {
		// Prepare to receive tactile sensor frame (Push or Pull Mode)
		setFramerate(framerate);
	}

	if(startRecording) {
		enableRecording();
	}

	current_time.StoreNow();
	last_time = current_time;
	last_time_temperature = current_time;

	grabber_thread = boost::thread(boost::bind(&FrameGrabberDSA::execute, this));
}


void FrameGrabberDSA::execute() {
	printf("Entering DSA thread execution\n");

	nb_frames = 0;
	nb_errors = 0;
	while(true) {
		nb_frames++;
		try	{

			conditionalBreakpoint(); // Check preconditions for next iteration

			if(request_single_frames) {
				start_time.StoreNow();
				fprintf(stderr, "   <<<<<<<< request a single frame >>>>>>>\n\n");
				ts->SetFramerateRetries(0, true, true, 3); // try to request a single frame (at most 3 times)
			}

			ts->UpdateFrame(); // Reads frame from sensor controller and stores it in library's frame buffer

			current_time.StoreNow();

			//std::cout << "    Actual framerate=" << std::setprecision(2) << std::fixed << (1.0/last_time.Elapsed(current_time)) << "Hz nb_frames=" << nb_frames << " nb_errors=" << nb_errors << " (" << ((100.0*nb_errors)/nb_frames) << "%)\n";
			//std::cout.flush();
			last_time = current_time;

			if(recording) {
				// Store frame permanently
				frameManager->addTSFrame();
			}

			frameManager->setLiveFrame(); // Copy received frame for rendering and further processing

			if(request_single_frames) {
				remaining_time = frame_interval - (start_time.Elapsed());
				if(remaining_time > 0.0 ) {
					boost::this_thread::sleep(boost::posix_time::milliseconds(remaining_time*1000));
				}
			}
		} catch (cDSAException* e) {
			nb_errors++;
			std::cerr << "Caught and ignored cDSAException: " << e->what() << " nb_errors=" << nb_errors << "  Frame " << nb_frames << "\n";
			delete e;
		}
	}
	printf("Exiting DSA thread\n");
}


void FrameGrabberDSA::conditionalBreakpoint() {

	// Grabber thread blocks while paused
	boost::mutex::scoped_lock pause_lock(paused_mutex);

	// Disable Push Mode
	if(paused && !request_single_frames) {
		ts->SetFramerateRetries(0, true, false, 3, true); // Disable DSA push-mode
	}

	while(paused) { // Loop to catch spurious wake-ups
		paused_changed.wait(pause_lock); // wait() unlocks automatically
	}

	// Prepare to receive tactile sensor frame (Push or Pull Mode)
	setFramerate(framerate); // Restore previous framerate
}


void FrameGrabberDSA::pause() {
	// Change pause state as well as condition variable
	boost::mutex::scoped_lock lock_pause(paused_mutex);
	paused = true;
	paused_changed.notify_one();
	lock_pause.unlock(); // Should be unnecessary as lock goes out of scope

	// Make sure all transmissions on the serial port have stopped
	boost::this_thread::sleep(boost::posix_time::milliseconds(200));
}


void FrameGrabberDSA::resume() {
	assert(paused == true && "Frame grabber has to be paused first to be resumed");
	// Change pause state as well as condition variable
	boost::mutex::scoped_lock lock(paused_mutex);
	paused = false;
	paused_changed.notify_one();
	lock.unlock(); // Should be unnecessary as lock goes out of scope
}

void FrameGrabberDSA::finish() {
	pause();
	grabber_thread.interrupt(); // Ask thread to stop
	grabber_thread.join(); // Wait for it
}


void FrameGrabberDSA::enableRecording() {
	assert(frameManager != NULL && "Unable to start recording without a FrameManager instance");
	boost::mutex::scoped_lock lock(recording_mutex);
	recording = true;
	printf("DSA Recording enabled\n");
}


void FrameGrabberDSA::suspendRecording() {
	boost::mutex::scoped_lock lock(recording_mutex);
	recording = false;
	printf("DSA Recording suspended\n");
}


bool FrameGrabberDSA::isRunning() {
	return grabber_thread.try_join_for(boost::chrono::milliseconds(0)); // *ugly hack*
}


bool FrameGrabberDSA::isCapturing() {
	return !paused;
}


bool FrameGrabberDSA::isRecording() {
	return recording;
}
