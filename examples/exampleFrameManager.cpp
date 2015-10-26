#include <cstdlib>
#include <cstdio>

#include <iostream>
#include <fstream>
#include <iomanip>

#include <string>
#include <vector>
#include <cmath>

#include <stdint.h>
#include <sys/time.h>

#include <boost/program_options.hpp>

#include "framemanager.h"
#include "utils.h"

using namespace std;

FrameManager frameManager;
FrameProcessor* frameProcessor;

uint numFrames;
uint numCells;
uint64_t starttime;
uint64_t stoptime;
double fps;

int main(int argc, char* argv[]) {

    // Check the number of parameters
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " profile.dsa" << std::endl;
        return EXIT_FAILURE;
    }

    std::string profileName = argv[1];

    frameManager.loadFrames(profileName);
    frameProcessor = frameManager.getFrameProcessor();
    numFrames = frameManager.getFrameCountTS();
    numCells = frameManager.getSensorInfo().nb_cells;
    starttime = frameManager.getFrame(0)->timestamp;
    stoptime = frameManager.getFrame(numFrames-1)->timestamp;
    fps = static_cast<double>(numFrames) / (static_cast<double>(stoptime-starttime)/1000.0);
    printf("\n======================\nProfile summary:\n======================\n");
    printf("Frames: %d, start: %lld, stop: %lld, fps(average): %f\n", numFrames, starttime, stoptime, fps);

    printf("Done!\n");

    return EXIT_SUCCESS;
}
