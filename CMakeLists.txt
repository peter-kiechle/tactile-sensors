cmake_minimum_required(VERSION 2.8.11)

# Project name
project(masterthesis)

# Prevent in-source builds
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_VERBOSE_MAKEFILE ON)


###########################
# Find necessary packages
###########################

# SDH
find_library( SDH_LIBRARY SDHLibrary-CPP REQUIRED )
set( SDH_INCLUDE "/usr/include/sdh" CACHE FILEPATH "SDH Library include directory" )
include_directories( ${SDH_INCLUDE} )

# Python
find_package( PythonLibs 2.7 REQUIRED )
include_directories( ${PYTHON_INCLUDE_DIR} )

# Boost
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost COMPONENTS python thread system chrono filesystem iostreams program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )

# Boost.Numpy
find_library(BOOST_NUMPY_LIBRARY boost_numpy REQUIRED)

# Eigen 3
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# pkg-config (in order to find GTKMM)
find_package(PkgConfig REQUIRED)

# GTKMM
pkg_check_modules(GTKMM gtkmm-2.4) # Defines GTKMM_INCLUDE_DIRS, GTKMM_LIBRARY_DIRS and GTKMM_LIBRARIES
include_directories(${GTKMM_INCLUDE_DIRS})

# GTKGLEXTMM (GTKMM OpenGL extensions)
include(FindPkgConfig)
pkg_search_module(GTKGLEXTMM gtkglextmm-x11-1.2) # Defines GTKGLEXTMM_INCLUDE_DIRS, GTKGLEXTMM_LIBRARY_DIRS and GTKGLEXTMM_LIBRARIES
include_directories(${GTKGLEXTMM_INCLUDE_DIRS})


###############################################################
# Default Output directories (set before the add_subdirectory)
###############################################################
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )


##########################
# Process subdirectories
##########################
add_subdirectory(src)
add_subdirectory(examples)


##################
# Output settings
##################

# Cmake configuration
message("PROJECT_NAME=${PROJECT_NAME}")
message("PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
message("PROJECT_BINARY_DIR=${PROJECT_BINARY_DIR}")
message("CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
message("CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}")
message("CMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message("CMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

# Include and library paths
message(SDH_LIBRARY=${SDH_LIBRARY})
message(SDH_INCLUDE=${SDH_INCLUDE})
message(Boost_INCLUDE_DIR=${Boost_INCLUDE_DIR})
message(OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS})
message(EIGEN_INCLUDE_DIRS=${EIGEN_INCLUDE_DIRS})
message(PYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR})
message(BOOST_NUMPY_LIBRARY=${BOOST_NUMPY_LIBRARY})
message(GTKMM_INCLUDE_DIRS=${GTKMM_INCLUDE_DIRS})
message(GTKMM_LIBRARIES=${GTKMM_LIBRARIES})
message(GTKGLEXTMM_INCLUDE_DIRS=${GTKGLEXTMM_INCLUDE_DIRS})
message(GTKGLEXTMM_LIBRARIES=${GTKGLEXTMM_LIBRARIES})
