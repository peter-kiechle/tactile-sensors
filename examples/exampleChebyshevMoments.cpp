#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

#include <stdint.h>
#include <sys/time.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tuple/tuple.hpp>

#include "framemanager.h"
#include "chebyshevMoments.h"
#include "utils.h"
#include "colormap.h"

using namespace cv;
using namespace std;


RGB determineColor(Colormap &colormap, float value) {
    if(utils::almostEqual(value, 0.0, 4)) { // No signal
        RGB color(0.3, 0.3, 0.3); // Light Grey
        return color;
    } else {
        RGB color = colormap.getColorFromTable(static_cast<int>(value+0.5));
        return color;
    }
}

uint numFrames;
uint numCells;
uint64_t starttime;
uint64_t stoptime;
double fps;

std::string filename;
bool isPressureProfile;
uint matrixID = 0;
std::vector<uint> frameList;

int pmax = 5; // maximum moment order

uint startFrame = 0;
uint stopFrame = 0;
double speedFactor = 1.0;

FrameManager frameManager;
FrameProcessor* frameProcessor;

bool parseCommandLine(int argc, char** argv) {
    namespace po = boost::program_options;

    try
    {
        po::options_description description("Usage");
        description.add_options() // Note: The order of options is implicitly defined
                                  ("help,h", "Display this information")
                                  ("input-file,i", po::value<std::string>()->required(), "Input pressure profile")
                                  ("matrixID,m", po::value<uint>(), "MatrixID")
                                  ("frameID,f", po::value<std::vector<uint> >(&frameList)->multitoken(), "List of frames")
                                  ("pmax,pm", po::value<uint>(), "Moment order")
                                  ;

        // Missing option name is mapped to "input-file"
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
            filename = vm["input-file"].as<std::string>();
            std::cout << "filename: " << filename << std::endl;
        }

        if(vm.count("matrixID")) {
            matrixID = vm["matrixID"].as<uint>();
            std::cout << "matrix: " << matrixID << std::endl;
        }

        if(vm.count("frameID")) {
            frameList = vm["frameID"].as<std::vector<uint> >();
            std::cout << "frame list: ";
            for(uint i = 0; i < frameList.size(); i++) {
                std::cout << frameList[i] << " ";
            }
            std::cout << std::endl;
        }

        if(vm.count("pmax")) {
            pmax = vm["pmax"].as<uint>();
            std::cout << "pmax: " << pmax << std::endl;
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

    // Primitive check if specified file is a pressure profile
    boost::filesystem::path path(filename);
    std::string extension = path.extension().string();

    if(extension == ".dsa") {
        isPressureProfile = true;
        frameManager.loadFrames(filename);
        frameProcessor = frameManager.getFrameProcessor();
        numFrames = frameManager.getFrameCountTS();
        starttime = frameManager.getFrame(0)->timestamp;
        stoptime = frameManager.getFrame(numFrames-1)->timestamp;
        fps = static_cast<double>(numFrames) / (static_cast<double>(stoptime-starttime)/1000.0);
        printf("\n======================\nProfile summary:\n======================\n");
        printf("Frames: %d, start time: %lld, stop time: %lld, duration: %.2f s, fps(average): %.2f\n", numFrames, starttime, stoptime, (stoptime-starttime)/1000.0, fps);
    } else {
        isPressureProfile  = false;
    }


    Chebyshev CM;
    array_type T_pq_doubleprime(boost::extents[pmax][pmax]);

    cv::Mat frame = Mat::zeros(14, 6, CV_32F);

    // Load frame from pressure profile or PNG-file
    if(isPressureProfile) {
        printf("matrixID: %d, frameID: %d\n", matrixID, frameList[0]);
        TSFrame* tsFrame = frameManager.getFrame(frameList[0]);
        matrixInfo &matrixInfo = frameManager.getMatrixInfo(matrixID);
        frame = cv::Mat( matrixInfo.cells_y, matrixInfo.cells_x, CV_32F, tsFrame->cells.data()+matrixInfo.texel_offset).clone(); // Copying

        // Scale such that highest intensity is 1.0
        //cv::normalize(frame, frame, 0.0, 1.0, NORM_MINMAX, CV_32F);

        // Scale [0..1] assuming max value is 255
        frame.convertTo(frame, CV_32F, 1.0/4096.0);

    } else {
        frame = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

        if(frame.data == NULL) {
            fprintf(stderr, "Error opening image: %s\n", filename.c_str());
            exit(EXIT_FAILURE);
        }

        // Scale such that highest intensity is 1.0
        //cv::normalize(frame, frame, 0.0, 1.0, NORM_MINMAX, CV_32F);

        // Scale [0..1] assuming max value is 255
        frame.convertTo(frame, CV_32F, 1.0/255.0);
    }

    CM.computeInvariants(frame, pmax, T_pq_doubleprime);

    printf("---------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("Translation and rotation invariant Chebyshev moments:\n");
    printf("---------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // 1D
    for(int p = 0; p < pmax; p++) {
        for(int q = 0; q < pmax; q++) {
            //printf("% 15.8f ", T_pq_doubleprime[p][q]);
            printf("% 7.3f ", T_pq_doubleprime[p][q]);
        }
        printf("\n");
    }
    printf("\n");

    // Compute reconstruction from Chebyshev moments
    int N = frame.rows;
    cv::Mat frame_reconstructed = cv::Mat(N, N, CV_32F, cv::Scalar(0.0));
    CM.computeReconstruction(frame_reconstructed);

    frame_reconstructed.convertTo(frame_reconstructed, CV_8UC1, 255.0); // [0..255]

    // Save reconstructed file
    boost::filesystem::path path_outfile(filename);
    std::string outbasename = path_outfile.stem().string() + "_reconstructed.png";
    boost::filesystem::path full_path_outfile = path_outfile.parent_path() / boost::filesystem::path(outbasename);
    std::string outfile = full_path_outfile.string();
    try {
        cv::imwrite(outfile, frame_reconstructed);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Error saving file: %s, %s\n", outfile.c_str(), ex.what());
    }
    fprintf(stdout, "Saved file %s.\n", outfile.c_str());

    // Display frames
    int window_width = 400;
    int window_height = 400;
    int offset_x = 200;
    int offset_y = 200;

    cv::namedWindow("Frame", CV_WINDOW_NORMAL); // CV_WINDOW_AUTOSIZE CV_WINDOW_NORMAL
    resizeWindow("Frame", window_width, window_height);
    cv::moveWindow("Frame", 0*window_width + offset_x, offset_y);
    cv::imshow("Frame", frame);

    cv::namedWindow("Reconstruction", CV_WINDOW_NORMAL); // CV_WINDOW_AUTOSIZE CV_WINDOW_NORMAL
    resizeWindow("Reconstruction", window_width, window_height);
    cv::moveWindow("Reconstruction", 1*window_width + offset_x, offset_y);
    cv::imshow("Reconstruction", frame_reconstructed);
    cv::waitKey(0);

    return EXIT_SUCCESS;
}
