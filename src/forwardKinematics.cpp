#include <iostream>
#include <iomanip>

#include "forwardKinematics.h"
#include "utils.h"

using namespace std;

ForwardKinematics::ForwardKinematics() {

	// See thesis for details
	m_taxel_width = 3.4;
	m_matrix_width = 6*m_taxel_width;

	m_l1 = 16.7;
	m_l2 = 86.5;
	m_l3 = 68.5;

	m_s1 = 17.5;
	m_s3 = 17.5;

	m_a1 = 4.1;
	m_a2 = 4.95;

	m_d1 = 15.4;
	m_d2 = 14.8;

	m_R = 58.0; // Haase claims 60.0

	// Create static transformation matrices
	createTransformationTable_proximal();
	createTransformationTable_distal();
}

ForwardKinematics::~ForwardKinematics() {}


void ForwardKinematics::setAngles(std::vector<double> &all_angles) {
	// Create dynamic transformation matrices
	createTransformationMatrices_f0(all_angles[0], all_angles[1], all_angles[2]);
	createTransformationMatrices_f1(all_angles[3], all_angles[4]);
	createTransformationMatrices_f2(all_angles[0], all_angles[5], all_angles[6]);
}


void ForwardKinematics::createTransformationMatrix(double d_n, double Theta_n, double r_n, double alpha_n, cv::Matx44d& T) {

	T = cv::Matx44d( cos(Theta_n), -sin(Theta_n)*cos(alpha_n),  sin(Theta_n)*sin(alpha_n), r_n*cos(Theta_n),
			         sin(Theta_n),  cos(Theta_n)*cos(alpha_n), -cos(Theta_n)*sin(alpha_n), r_n*sin(Theta_n),
			         0.0,           sin(alpha_n),               cos(alpha_n),              d_n,
			         0.0,           0.0,                        0.0,                       1.0 );
}


void ForwardKinematics::createTransformationMatrices_f0(double phi0, double phi1, double phi2) {

	// Denavit-Hartenberg parameters
	double distance[3] = { m_l1, 0.0, 0.0 };
	double Theta[3] = { -phi0, phi1 - utils::degToRad(90.0), phi2 };
	double r[3] = { 0.0, m_l2, m_l3};
	double alpha[3] =  { utils::degToRad(-90.0), 0.0, 0.0 };

	// Finger transformation matrices
	cv::Matx44d T01_f0;
	cv::Matx44d T12_f0;
	cv::Matx44d T23_f0;
	createTransformationMatrix(distance[0], Theta[0], r[0], alpha[0], T01_f0);
	createTransformationMatrix(distance[1], Theta[1], r[1], alpha[1], T12_f0);
	createTransformationMatrix(distance[2], Theta[2], r[2], alpha[2], T23_f0);

	// Apply finger offset (equilateral triangle relations)
	double tx = 19.05255888325765; // 1/3 * h_tiangle = 1/3 * (1/2*sqrt(3)*d);
	double ty = -33.0; // -0.5 * d;
	double tz = 0.0;

	// Finger frame
	cv::Matx44d T00_f0 = cv::Matx44d( -1.0, 0.0, 0.0, tx,
			                           0.0, -1.0, 0.0, ty,
			                           0.0, 0.0, 1.0,  tz,
			                           0.0, 0.0, 0.0, 1.0);

	T01_f0 = T00_f0 * T01_f0; // Frame after rotational joint
	m_T02_f0 = T01_f0 * T12_f0; // Base frame for proximal sensor matrix
	m_T03_f0 = m_T02_f0 * T23_f0; // Base frame for distal sensor matrix (End effector, i.e. fingertip)
}

void ForwardKinematics::createTransformationMatrices_f1(double phi3, double phi4) {

	// Denavit-Hartenberg parameters
	double distance[3] = { m_l1, 0.0, 0.0 };
	double Theta[3] = { utils::degToRad(180.0), phi3 - utils::degToRad(90.0), phi4 }; // phi0 is stiffened
	double r[3] = { 0.0, m_l2, m_l3};
	double alpha[3] =  { utils::degToRad(-90.0), 0.0, 0.0 };

	// Finger transformation matrices
	cv::Matx44d T01_f1;
	cv::Matx44d T12_f1;
	cv::Matx44d T23_f1;
	createTransformationMatrix(distance[0], Theta[0], r[0], alpha[0], T01_f1);
	createTransformationMatrix(distance[1], Theta[1], r[1], alpha[1], T12_f1);
	createTransformationMatrix(distance[2], Theta[2], r[2], alpha[2], T23_f1);

	// Apply finger offset (Equilateral triangle relations)
	double tx = -38.1051177665153; // -2/3 * h_tiangle = -2/3 * (1/2*sqrt(3)*d);
	double ty = 0.0;
	double tz = 0.0;

	// Finger frame
	cv::Matx44d T00_f1 = cv::Matx44d( -1.0, 0.0, 0.0, tx,
	                                   0.0, -1.0, 0.0, ty,
	                                   0.0, 0.0, 1.0,  tz,
	                                   0.0, 0.0, 0.0, 1.0);

	T01_f1 = T00_f1 * T01_f1; // Frame after rotational joint
	m_T02_f1 = T01_f1 * T12_f1; // Base frame for proximal sensor matrix
	m_T03_f1 = m_T02_f1 * T23_f1; // Base frame for distal sensor matrix (End effector, i.e. fingertip)
}


void ForwardKinematics::createTransformationMatrices_f2(double phi0, double phi5, double phi6) {

	// Denavit-Hartenberg parameters
	double distance[3] = { m_l1, 0.0, 0.0 };
	double Theta[3] = { phi0, phi5 - utils::degToRad(90.0), phi6 };
	double r[3] = { 0.0, m_l2, m_l3};
	double alpha[3] =  { utils::degToRad(-90.0), 0.0, 0.0 };

	// Finger transformation matrices
	cv::Matx44d T01_f2;
	cv::Matx44d T12_f2;
	cv::Matx44d T23_f2;
	createTransformationMatrix(distance[0], Theta[0], r[0], alpha[0], T01_f2);
	createTransformationMatrix(distance[1], Theta[1], r[1], alpha[1], T12_f2);
	createTransformationMatrix(distance[2], Theta[2], r[2], alpha[2], T23_f2);

	// Apply finger offset (Equilateral triangle relations)
	double tx = 19.05255888325765; // 1/3 * h_tiangle = 1/3 * (1/2*sqrt(3)*d);
	double ty = 33.0; // 0.5 * d;
	double tz = 0.0;

	// Finger frame
	cv::Matx44d T00_f2 = cv::Matx44d( -1.0, 0.0, 0.0, tx,
	                                   0.0, -1.0, 0.0, ty,
	                                   0.0, 0.0, 1.0,  tz,
	                                   0.0, 0.0, 0.0, 1.0);

	T01_f2 = T00_f2 * T01_f2; // Frame after rotational joint
	m_T02_f2 = T01_f2 * T12_f2; // Base frame for proximal sensor matrix
	m_T03_f2 = m_T02_f2 * T23_f2; // Base frame for distal sensor matrix (End effector, i.e. fingertip)
}


cv::Matx44d ForwardKinematics::createTransformationMatrix_proximal() {

	// Transformation within the sensor matrix
	double sx =  m_taxel_width;
	double sy = -m_taxel_width;
	double tx = m_taxel_width/2.0 - m_matrix_width/2.0;
	double ty = 13*m_taxel_width;

	// Translation relative to O2
	double tx2 = -m_l2 + m_s1 + m_a1;
	double ty2 = m_d1;

	return cv::Matx44d( sy, 0.0, 0.0, ty+tx2,
	                    0.0, 1.0, 0.0,  ty2,
	                    0.0, 0.0,  sx,   tx,
	                    0.0, 0.0, 0.0,  1.0);
}


cv::Matx44d ForwardKinematics::createTransformationMatrix_distal(double y) {

	double sx, sy, tx, ty, tx3, ty3;

	if(y > 8.5) { // Planar part

		// Transformation within sensor matrix
		sx = m_taxel_width;
		sy = -m_taxel_width;
		tx = m_taxel_width/2.0 - m_matrix_width/2.0;
		ty = 12*m_taxel_width;

		// Translation relative to O3
		tx3 = -m_l3 + m_s3 + m_a2;
		ty3 = m_d2;

	} else { // Curved part

		// Transformation within sensor matrix
		sx = m_taxel_width;
		sy = 1.0;
		tx = m_taxel_width/2.0 - m_matrix_width/2.0;
		ty = (m_R * sin( ((8.5-y)*m_taxel_width) / m_R )) + 3.5*m_taxel_width - y;

		// Translation relative to O3
		tx3 = -m_l3 + m_s3 + m_a2;
		ty3 = m_d2 - (m_R - m_R * cos( ((8.5-y)*m_taxel_width) / m_R));
	}

	return cv::Matx44d(  sy, 0.0, 0.0, ty+tx3,
	                     0.0, 1.0, 0.0,  ty3,
	                     0.0, 0.0,  sx,   tx,
	                     0.0, 0.0, 0.0,  1.0);
}


void ForwardKinematics::createTransformationTable_proximal() {
	m_P_prox = createTransformationMatrix_proximal();
}


void ForwardKinematics::createTransformationTable_distal() {
	m_P_dist.resize(13);
	for(int y = 0; y < 13; y++) {
		m_P_dist[y] = createTransformationMatrix_distal(static_cast<double>(y));
	}
}


cv::Matx44d ForwardKinematics::computeTransformationMatrixTaxelXYZ(int m, int y) {
	cv::Matx44d T_total;

	// Combine dynamic transformation matrices of finger joints with static sensor matrix offset
	if(m == 0) { // Finger 0: Proximal
		T_total = m_T02_f0 * m_P_prox;
	}
	else if(m == 1) { // Finger 0: Distal
		T_total = m_T03_f0 * m_P_dist[y];
	}
	else if (m == 2) { // Finger 1: Proximal
		T_total = m_T02_f1 * m_P_prox;
	}
	else if(m == 3) { // Finger 1: Distal
		T_total = m_T03_f1 * m_P_dist[y];
	}
	else if (m == 4) { // Finger 2: Proximal
		T_total = m_T02_f2 * m_P_prox;
	}
	else if(m == 5) { // Finger 2: Distal
		T_total = m_T03_f2 * m_P_dist[y];
	}
	return T_total;
}


cv::Matx44d ForwardKinematics::computeTransformationMatrixPointOnSensorPlaneXYZ(int m, double y) {

	cv::Matx44d T_total;
	double y_prime = y/3.4;

	// Combine dynamic transformation matrices of finger joints with dynamic sensor matrix coordinates
	if(m == 0) { // Finger 0: Proximal
		T_total = m_T02_f0 * createTransformationMatrix_proximal();
	}
	else if(m == 1) { // Finger 0: Distal
		T_total = m_T03_f0 * createTransformationMatrix_distal(y_prime);
	}
	else if (m == 2) { // Finger 1: Proximal
		T_total = m_T02_f1 * createTransformationMatrix_proximal();
	}
	else if(m == 3) { // Finger 1: Distal
		T_total = m_T03_f1 * createTransformationMatrix_distal(y_prime);
	}
	else if (m == 4) { // Finger 2: Proximal
		T_total = m_T02_f2 * createTransformationMatrix_proximal();
	}
	else if(m == 5) { // Finger 2: Distal
		T_total = m_T03_f2 * createTransformationMatrix_distal(y_prime);
	}
	return T_total;
}


std::vector<double> ForwardKinematics::GetTaxelXYZ(int m, int x, int y) {

	cv::Matx44d T_total = computeTransformationMatrixTaxelXYZ(m, y);

	// Map cell index to point in 3D sensor matrix space
	cv::Vec4d p = cv::Vec4d(y, 0.0, x, 1.0);

	// Transform point
	cv::Vec4d p_transformed = T_total * p;

	std::vector<double>coordinate(3);
	coordinate[0] = p_transformed[0];
	coordinate[1] = p_transformed[1];
	coordinate[2] = p_transformed[2];

	return coordinate;
}


std::vector<double> ForwardKinematics::GetPointOnSensorPlaneXYZ(int m, double x, double y) {

	cv::Matx44d T_total = computeTransformationMatrixPointOnSensorPlaneXYZ(m, y);

	double x_prime = x/3.4;
	double y_prime = y/3.4;

	// Map cell index to point in 3D sensor matrix space
	cv::Vec4d p = cv::Vec4d(y_prime, 0.0, x_prime, 1.0);

	// Transform point
	cv::Vec4d p_transformed = T_total * p;

	std::vector<double>coordinate(3);
	coordinate[0] = p_transformed[0];
	coordinate[1] = p_transformed[1];
	coordinate[2] = p_transformed[2];

	return coordinate;
}
