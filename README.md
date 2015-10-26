# DSA-Konqueror and libframemanager
Tactile sensor frame manager for the SCHUNK Dextrous Hand 2.0 (SDH 2.0)

# What is this?
This software package was created during the course of my master thesis "Evaluation of Tactile Sensors"
at the Intelligent and Interactive Systems group of the Institute of Computer Science at the University of Innsbruck.
My supervisor was Professor Justus Piater, Ph.D.

Libframemanager consists of a frame manager and two separate frame grabbers. 
One grabber requests and processes the temperatures and axis angles received from 
the SDH-2 while the other captures the tactile sensor frames from the DSA controller.
DSA Konqueror, not to be confused with DSA Explorer by Weiss Robotics, is a graphical user interface or GUI
to control the SDH-2, i.e. perform grasps and display the tactile sensor readings.
It simplifies the capturing and recording of pressure profiles and offers real-time visualization as well as offline processing. In addition, there are Python bindings to the most frequently used functionalities.
See master thesis for details.


# Compile with:
cd build
cmake ..
make VERBOSE=1

Dependencies: see "dependencies" folder

  C++:
    SDH Library (patched version, see dependencies/dsa_patch)
    OpenCV
    Eigen 3
    Boost
    Boost.NumPy (extension for Boost.Python that adds NumPy support, see dependencies/Boost.NumPy)
    Gtkmm 2.4 (GUI only)
    GtkGLExtmmb (OpenGL extension)

  Python 2.7:
    numpy
    scipy
    sklearn 
    cairo
    PIL


# License
DSA-Konqueror and libframemanager
Copyright (C) 2015  Peter Kiechle, peter@kiechle-pfronten.de

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
