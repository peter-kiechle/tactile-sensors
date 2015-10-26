#######################
# Build shared library
#######################

# Make DEFAULT the default build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE DEFAULT)
endif(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

# Compiler flags
set(CMAKE_CXX_FLAGS "-std=c++0x -Wall -fms-extensions") # Common to all build types
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_ESD_CAN=0 -DWITH_PEAK_CAN=0 -DOSNAME_LINUX=1") # Needed for sdh-library
set(CMAKE_CXX_FLAGS_DEFAULT "-O2 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DEIGEN_NO_DEBUG")

# Target specific compile flags
#set_target_properties(libchicken PROPERTIES
#  COMPILE_FLAGS "-DC_BUILDING_LIBCHICKEN ${SHARED_FLAGS}"
#)


#########################
# Frame Manager Library
#########################
add_library(framemanager SHARED framemanager.cpp
                                framegrabberSDH.cpp
                                framegrabberDSA.cpp
                                frameprocessor.cpp
                                slipdetection.cpp
                                utils.cpp
                                sdhoptions.cpp )

target_link_libraries(framemanager ${SDH_LIBRARY} ${Boost_LIBRARIES} ${OpenCV_LIBS} )


################################
# Feature Extraction Library
################################
add_library(featureextraction SHARED chebyshevMoments.cpp 
                              forwardKinematics.cpp
                              featureExtraction.cpp )

target_link_libraries(featureextraction framemanager )



#############################################
# Boost Python / Boost Numpy wrapper library
#############################################
add_library(framemanager_python SHARED framemanager_python.cpp)
target_link_libraries(framemanager_python framemanager featureextraction ${BOOST_NUMPY_LIBRARY} )
set_target_properties(framemanager_python PROPERTIES PREFIX "" ) # No "lib" prefix to python module name



###############
# DSAKonqueror
###############
add_executable(DSA-Konqueror
               calibration.cpp
               camera.cpp
               chebyshevMoments.cpp
               colormap.cpp
               controller.cpp
               extension.cpp
               featureExtraction.cpp
               forwardKinematics.cpp
               framegrabberDSA.cpp
               framegrabberSDH.cpp
               framemanager.cpp
               frameprocessor.cpp
               guiChart.cpp
               guiController.cpp
               guiGraph.cpp               
               guiMain.cpp
               guiRenderer2D.cpp
               guiRenderer3D.cpp
               guiRenderer.cpp
               guiScreenshot.cpp
               guiSeekbar.cpp
               guiSlipDetection.cpp
               guiSlipDetectionMultiPlot.cpp
               guiTools.cpp
               guiTreeView.cpp
               mainKonqueror.cpp
               sdhoptions.cpp
               slipdetection.cpp
               utils.cpp
              )

target_link_libraries(DSA-Konqueror ${SDH_LIBRARY} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${GTKMM_LIBRARIES} ${GTKGLEXTMM_LIBRARIES} ${PYTHON_LIBRARIES})





#####################
# Output directories
#####################
set_target_properties(framemanager framemanager_python featureextraction
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

set_target_properties(DSA-Konqueror
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/DSA-Konqueror"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
)



#INSTALL(TARGETS framemanager framemanager_python gaussian_python
#  RUNTIME DESTINATION bin
#  LIBRARY DESTINATION lib
#  ARCHIVE DESTINATION lib
#)
