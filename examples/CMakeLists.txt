####################
# Build executables
####################

# Make DEFAULT the default build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE DEFAULT)
endif(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS "-std=c++0x -Wall -fms-extensions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_ESD_CAN=0 -DWITH_PEAK_CAN=0 -DOSNAME_LINUX=1")
set(CMAKE_CXX_FLAGS_DEFAULT "-O2 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")


# Make sure the compiler can find include files
#include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

# Make sure the linker can find the library once it is built
#link_directories (${PROJECT_BINARY_DIR}/lib) 


##############################
# Basic frame manager example
##############################
add_executable(exampleFrameManager 
               exampleFrameManager.cpp
               ${PROJECT_SOURCE_DIR}/src/utils.cpp
               ${PROJECT_SOURCE_DIR}/src/slipdetection.cpp
               ${PROJECT_SOURCE_DIR}/src/forwardKinematics.cpp
               ${PROJECT_SOURCE_DIR}/src/framemanager.cpp
               ${PROJECT_SOURCE_DIR}/src/frameprocessor.cpp
               ${PROJECT_SOURCE_DIR}/src/framegrabberSDH.cpp
               ${PROJECT_SOURCE_DIR}/src/framegrabberDSA.cpp
)

target_link_libraries(exampleFrameManager ${SDH_LIBRARY} ${Boost_LIBRARIES} ${OpenCV_LIBS})


############################
# Feature extraction exmple
############################
add_executable(exampleFeatureExtraction
               exampleFeatureExtraction.cpp
               ${PROJECT_SOURCE_DIR}/src/utils.cpp
               ${PROJECT_SOURCE_DIR}/src/slipdetection.cpp
               ${PROJECT_SOURCE_DIR}/src/forwardKinematics.cpp
               ${PROJECT_SOURCE_DIR}/src/framemanager.cpp
               ${PROJECT_SOURCE_DIR}/src/frameprocessor.cpp
               ${PROJECT_SOURCE_DIR}/src/framegrabberSDH.cpp
               ${PROJECT_SOURCE_DIR}/src/framegrabberDSA.cpp
               ${PROJECT_SOURCE_DIR}/src/featureExtraction.cpp
               ${PROJECT_SOURCE_DIR}/src/chebyshevMoments.cpp
              )

target_link_libraries(exampleFeatureExtraction ${SDH_LIBRARY} ${Boost_LIBRARIES} ${OpenCV_LIBS})


##################
# Extended exmple
##################
add_executable(exampleChebyshevMoments
               exampleChebyshevMoments.cpp
               ${PROJECT_SOURCE_DIR}/src/utils.cpp
               ${PROJECT_SOURCE_DIR}/src/slipdetection.cpp
               ${PROJECT_SOURCE_DIR}/src/forwardKinematics.cpp
               ${PROJECT_SOURCE_DIR}/src/framemanager.cpp
               ${PROJECT_SOURCE_DIR}/src/frameprocessor.cpp
               ${PROJECT_SOURCE_DIR}/src/framegrabberSDH.cpp
               ${PROJECT_SOURCE_DIR}/src/framegrabberDSA.cpp
               ${PROJECT_SOURCE_DIR}/src/featureExtraction.cpp
               ${PROJECT_SOURCE_DIR}/src/chebyshevMoments.cpp
               ${PROJECT_SOURCE_DIR}/src/colormap.cpp
              )

target_link_libraries(exampleChebyshevMoments ${SDH_LIBRARY} ${Boost_LIBRARIES} ${OpenCV_LIBS})



# Where to put the compiled target files
set_target_properties(exampleFrameManager exampleFeatureExtraction exampleChebyshevMoments
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)



