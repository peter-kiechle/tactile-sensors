#ifndef FORWARDKINEMATICS_H_
#define FORWARDKINEMATICS_H_

#include <vector>
#include <opencv2/core/core.hpp>

/// Lookup table of transformation matrices
typedef std::vector<cv::Matx44d> Table_type;

/**
 * @class ForwardKinematics
 * @brief Implements forward kinematics for the tactile sensors.
 * @details The transformations are based on the classic Denavit-Hartenberg convention.
 *          The coordinate center is at gripper's base, i.e. in the middle of the equilateral triangle.
 *          See Figure 4.4: Isometric projection of my thesis for details.
 *          See featureExtraction.cpp for example usage.
 * @note Written for readability. Let's rely on the look-up table approach and the compiler's optimization.
 */
class ForwardKinematics {

public:
	/**
	 * Constructor automatically creates static transformation matrices
	 */
	ForwardKinematics();
	virtual ~ForwardKinematics();


	/**
	 * Initializes dynamic transformation matrices
	 * @param all_angles Angles [phi0 .. phi6] in Radians.
	 * @return void
	 */
	void setAngles(std::vector<double> &all_angles);


	/**
	 * Computes transformation matrix of taxel(_,y) on matrix m
	 * @param m Matrix index
	 * @param y y-coordinate on sensor matrix (only needed for distal matrix)
	 * @return Transformation matrix relative to hand's origin
	 */
	cv::Matx44d computeTransformationMatrixTaxelXYZ(int m, int y);


	/**
	 * Computes transformation matrix of position(_,y) on matrix m
	 * @param m Matrix index
	 * @param y y-coordinate on sensor matrix (only needed for distal matrix)
	 * @return Transformation matrix relative to hand's origin
	 */
	cv::Matx44d computeTransformationMatrixPointOnSensorPlaneXYZ(int m, double y);


	/**
	 * Computes Cartesian coordinates (in mm) of taxel(x,y) on matrix m
	 * @param m - Matrix index.
	 * @param x - x-coordinate on sensor matrix.
	 * @param y - y-coordinate on sensor matrix.
	 * @return Position [x,y,z] relative to hand's origin.
	 */
	std::vector<double> GetTaxelXYZ(int m, int x, int y);


	/**
	 * Computes Cartesian coordinates (in mm) of position(x,y) on matrix m
	 * @details (0.0, 0.0) refers to the center of the top-left taxel(0,0)
	 *          Meaningful values that actual lie on the sensor plane:
	 *          Proximal: ([-1.7, 18.7], [-1.7, 45.9])
	 *          Distal:   ([-1.7, 18.7], [-1.7, 42.5)
	 *
	 * @param m - Matrix index.
	 * @param x - x-coordinate on sensor matrix.
	 * @param y - y-coordinate on sensor matrix.
	 * @return Position [x,y,z] relative to hand's origin.
	 */
	std::vector<double> GetPointOnSensorPlaneXYZ(int m, double x, double y);

private:

	double m_l1;
	double m_l2;
	double m_l3;

	double m_s1;
	double m_s3;

	double m_a1;
	double m_a2;

	double m_d1;
	double m_d2;

	double m_R;
	double m_taxel_width;
	double m_matrix_width;


	cv::Matx44d m_P_prox; /// Proximal taxel transformation matrix (relative to sensor matrix base frame)
	Table_type m_P_dist; /// Table of distal taxel transformation matrices (relative to sensor matrix base frame)

	cv::Matx44d m_T02_f0; /// Transformation matrix: Finger 0, from \f$O_0\f$ to \f$O_2\f$
	cv::Matx44d m_T03_f0; /// Transformation matrix: Finger 0, from \f$O_0\f$ to \f$O_3\f$

	cv::Matx44d m_T02_f1; /// Transformation matrix: Finger 1, from \f$O_0\f$ to \f$O_2\f$
	cv::Matx44d m_T03_f1; /// Transformation matrix: Finger 1, from \f$O_0\f$ to \f$O_3\f$

	cv::Matx44d m_T02_f2; /// Transformation matrix: Finger 2, from \f$O_0\f$ to \f$O_2\f$
	cv::Matx44d m_T03_f2; /// Transformation matrix: Finger 2, from \f$O_0\f$ to \f$O_3\f$


	/**
	 *  Generates a generic transformation matrix from frame n-1 to frame n.
	 *  Follows the Denavit-Hartenberg convention.
	 * @param d_n Distance between \f$x_n\f$  and \f$x_n-1\f$ along \f$z_n-1\f$.
	 * @param Theta_n Rotation about \f$z_n-1\f$ (in Radians).
	 * @param r_n Radius of rotation about \f$z_n-1\f$.
	 * @param alpha_n Rotation about \f$x_n\f$ (in Radians).
	 * @param T Resulting transformation matrix.
	 * @return void
	 */
	void createTransformationMatrix(double d_n, double Theta_n, double r_n, double alpha_n, cv::Matx44d& T);


	/**
	 * Transforms proximal sensor matrix coordinates (x,y) to coordinate system O2 (x2, y2, z2).
	 * @return The proximal transformation matrix.
	 */
	cv::Matx44d createTransformationMatrix_proximal();


	/**
	 * Transforms distal sensor matrix coordinates (x,y) to coordinate system O3 (x3, y3, z3).
	 * @param y y-coordinate on distal sensor matrix.
	 * @return The distal transformation matrix.
	 */
	cv::Matx44d createTransformationMatrix_distal(double y);


	/**
	 * Creates a table of transformation matrices for each proximal taxel
	 * @return void
	 */
	void createTransformationTable_proximal();


	/**
	 * Creates a table of transformation matrices for each distal taxel
	 * @return void
	 */
	void createTransformationTable_distal();


	/**
	 * Generates transformation matrices from the hand's origin to the tip of Finger 0
	 * T02_f0 and T03_f0 correspond to the base frames of proximal and distal sensors
	 * @param phi0 Axis 0: Rotational angle
	 * @param phi1 Axis 1
	 * @param phi2 Axis 2
	 * @return void
	 */
	void createTransformationMatrices_f0(double phi0, double phi1, double phi2);


	/**
	 * Generates transformation matrices from the hand's origin to the tip of Finger 1
	 * T02_f1 and T03_f1 correspond to the base frames of proximal and distal sensors
	 * @param phi3 - Axis 3
	 * @param phi4 - Axis 4
	 * @return void
	 */
	void createTransformationMatrices_f1(double phi3, double phi4);


	/**
	 * Generates transformation matrices from the hand's origin to the tip of Finger 2
	 * T02_f2 and T03_f2 correspond to the base frames of proximal and distal sensors
	 * @param phi0 - Axis 0: Rotational angle
	 * @param phi5 - Axis 5
	 * @param phi6 - Axis 6
	 * @return void
	 */
	void createTransformationMatrices_f2(double phi0, double phi5, double phi6);

};

#endif /* FORWARDKINEMATICS_H_ */
