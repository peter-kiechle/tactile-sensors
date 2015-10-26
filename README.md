# DSA-Konqueror and libframemanager
Tactile sensor frame manager for the SCHUNK Dextrous Hand 2.0 (SDH 2.0)

![DSA Konqueror](src/doc/html/DSAKonqueror.png?raw=true "DSA Konqueror GUI")

Framework in action: https://www.youtube.com/channel/UCdTS8-T9_VR8UKo_lV9CZBA

# What is this?
This software package was created during the course of my ![master thesis "Evaluation of Tactile Sensors"](masterthesis.pdf "Evaluation of Tactile Sensors")
at the ![Intelligent and Interactive Systems](https://iis.uibk.ac.at/ "IIS") group of the Institute of Computer Science at the University of Innsbruck. 
My supervisor was Professor Justus Piater, Ph.D.

Libframemanager consists of a frame manager and two separate frame grabbers. 
One grabber requests and processes the temperatures and axis angles received from 
the SDH-2 while the other captures the tactile sensor frames from the DSA controller.
DSA Konqueror, not to be confused with DSA Explorer by Weiss Robotics, is a graphical user interface or GUI
to control the SDH-2, i.e. perform grasps and display the tactile sensor readings.
It simplifies the capturing and recording of pressure profiles and offers real-time visualization as well as offline processing. In addition, there are Python bindings to the most frequently used functionalities.

This project is not based on ROS. However, it offers several easily portable modules. <br />
Most notably:
<ul>
<li> <strong>Forward Kinematics</strong> of all taxels (following the Denavit-Hartenberg convention)</li>
<li> Translational and rotational <strong>Slip-Detection</strong> (based on convolution and the principal axis method)</li>
<li> Spatial,temporal and spatio-temporal <strong>filtering</strong> of the sensor signal </li>
<li> <strong>High-sensitivity mode</strong> for the SDH-2 (temperature calibrated trade-off between noise and sensitivity) </li>
<li> 2D Features: Translation and rotation invariant <strong>discrete Chebyshev moments</strong> </li>
<li> 3D Features: <strong>Object diameter</strong> and <strong>compressibility</strong> (minimal bounding sphere of active taxels)</li>
<li> <strong>Python bindings (NumPy) </strong> for convenient data analysis </li>
<li> <strong> Grasp-object classification</strong> based on scikit-learn</li>
</ul>
See master thesis for details.


# Compile with:
cd build <br />
cmake .. <br />
make VERBOSE=1 <br />


#Dependencies: see "dependencies" folder

  <strong>C++:</strong><br />
    SDH Library (patched version, see dependencies/dsa_patch) <br />
    OpenCV <br />
    Eigen 3 <br />
    Boost <br />
    Boost.NumPy (extension for Boost.Python that adds NumPy support, see dependencies/Boost.NumPy) <br />
    Gtkmm 2.4 (GUI only) <br />
    GtkGLExtmmb (OpenGL extension) <br />

  <strong>Python 2.7:</strong> <br />
    numpy <br />
    scipy <br />
    sklearn <br />
    cairo <br />
    PIL <br />


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
