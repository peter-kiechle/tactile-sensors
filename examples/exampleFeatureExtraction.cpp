#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

#include <stdint.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

// Boost
#include <boost/program_options.hpp>

#include "framemanager.h"
#include "utils.h"
#include "featureExtraction.h"

using namespace cv;
using namespace std;

FrameManager frameManager;
FrameProcessor* frameProcessor;

uint numFrames;
uint numCells;
uint64_t starttime;
uint64_t stoptime;
double fps;

std::string profileName;
uint frameID = 0;

bool parseCommandLine(int argc, char** argv) {
    namespace po = boost::program_options;

    try
    {
        po::options_description description("Usage");
        description.add_options() // Note: The order of options is implicitly defined
                                  ("help,h", "Display this information")
                                  ("input-file,i", po::value<std::string>()->required(), "Input pressure profile")
                                  ("frameID,f", po::value<uint>()->required(), "Frame ID")
                                  ;

        // "input-file" should be recognized by position
        po::positional_options_description pos;
        pos.add("input-file", -1); // number of consecutive values to be associated with the option

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(description).positional(pos).run(), vm);

        if(vm.count("help")) { // Print help before po::notify() complains about missing required variables
            std::cout << description << "\n";
            return false;
        }

        po::notify(vm); // Assign the specified variables, throws an error if required variables are missing

        if(vm.count("input-file")) {
            profileName = vm["input-file"].as<std::string>();
            std::cout << "filename: " << profileName << std::endl;
        }

        if(vm.count("frameID")) {
            frameID = vm["frameID"].as<uint>();
            std::cout << "frameID: " << frameID << std::endl;
        }
    }
    catch(std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return false;
    }
    catch(...) {
        std::cerr << "Unhandled Exception in main!" << "\n";
        return false;
    }

    return true;
}


int main(int argc, char* argv[]) {
    // Parse arguments
    bool result = parseCommandLine(argc, argv);
    if (!result) {
        return EXIT_FAILURE;
    }

    frameManager.loadFrames(profileName);
    frameProcessor = frameManager.getFrameProcessor();
    numFrames = frameManager.getFrameCountTS();
    numCells = frameManager.getSensorInfo().nb_cells;
    starttime = frameManager.getFrame(0)->timestamp;
    stoptime = frameManager.getFrame(numFrames-1)->timestamp;
    fps = static_cast<double>(numFrames) / (static_cast<double>(stoptime-starttime)/1000.0);
    printf("\n======================\nProfile summary:\n======================\n");
    printf("Frames: %d, start: %lld, stop: %lld, fps(average): %f\n", numFrames, starttime, stoptime, fps);


    // ------------------
    // Chebyshev moments
    // ------------------
    printf("Extracted features of frame %d\n\n", frameID);
    FeatureExtraction features(frameManager);

    int pmax = 5; // Max moment is (frameSize+1)/2 which is 7 for distal frames of size 6x13 (width is padded)

    for(uint m = 0; m < frameManager.getNumMatrices(); m++) {
        array_type T_pq_doubleprime = features.computeMoments(frameID, m, pmax);
        printf("shape: %d, %d, stride: %d, %d\n",  T_pq_doubleprime.shape()[0], T_pq_doubleprime.shape()[1], T_pq_doubleprime.strides()[0], T_pq_doubleprime.strides()[1]);
        printf("Rotation and translation invariant Chebyshev moments of matrix %d:\n", m);
        for(int p = 0; p < pmax; p++) {
            for(int q = 0; q < pmax; q++) {
                printf("% 15.8f ", T_pq_doubleprime[p][q]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");


    // ---------------------------------------
    // Standard deviation of intensity values
    // ---------------------------------------
    for(uint m = 0; m < frameManager.getNumMatrices(); m++) {
        double stdDev = features.computeStandardDeviation(frameID, m);
        printf("Standard deviation of matrix %d: %f\n", m, stdDev);
    }
    printf("\n");


    // ------------------------
    // Minimal bounding sphere
    // ------------------------
    JointAngleFrame *jointAngleFrame = frameManager.getCorrespondingJointAngle(frameID);

    std::vector<double> miniball = features.computeMiniball(frameID, jointAngleFrame->angles);

    printf("Diameter of minimal bounding sphere: %f\n", 2.0*miniball[3]);

    return EXIT_SUCCESS;
}
