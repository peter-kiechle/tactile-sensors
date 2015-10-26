#ifndef SLIPDETECTION_H_
#define SLIPDETECTION_H_

#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/bind.hpp>
#include <boost/tuple/tuple.hpp>

using namespace cv;
using namespace std;

/// Boost tuple return type for shape features
typedef boost::tuple<bool, cv::Point2d, cv::Point2d, double, double, double, double, double> shapeFeatures;

/// Boost tuple return type for rotational slip
typedef boost::tuple<bool, double, double, double, cv::Point2d, cv::Point2d, double, double, double, double> rotationResult;

/// Final return type struct: Boost tuples are limited to 10 elements, so this is why...
struct slipResult {
	bool successTranslation;
	bool successRotation;
	double slipVector_x;
	double slipVector_y;
	double slipVectorReference_x;
	double slipVectorReference_y;
	double slipAngle;
	double slipAngleReference;
	double orientation;
	double centroid_x;
	double centroid_y;
	double skew_x;
	double skew_y;
	double lambda1;
	double lambda2;
	double eccentricity;
	double compactness;
};

/**
 * @class Translation
 * @brief Implements the translational slip detection based on tracking the convolution matrix's center of gravity.
 */
class Translation {

private:
	int m_rows, m_cols;
	int m_rows_C, m_cols_C;
	cv::Mat m_A, m_B; 	// Index matrix of corresponding taxel position in C
	cv::Mat m_referenceFrame;
	cv::Mat m_previousFrame;
	cv::Point2d m_referenceDisplacement;
	cv::Point2d m_previousDisplacement;


	/**
	 * Flips matrix entries.
	 * Motivated by the difference between convolution and correlation.
	 * i.e. both integral transforms can be converted into one another by flipping one of the images in both directions.
	 * @param M The matrix.
	 * @return The flipped matrix.
	 */
	cv::Mat flip(cv::Mat M);


	/**
	 * Performs the actual convolution
	 * @param referenceFrame The reference tactile sensor matrix.
	 * @param currentFrame The current tactile sensor matrix.
	 * @param C The resulting convolution matrix.
	 */
	void convolve2d(cv::Mat &referenceFrame, cv::Mat& currentFrame, cv::Mat& C);

public:

	/**
	 * Constructor.
	 * Calls init()
	 * @param cols Tactile sensor width.
	 * @param rows Tactile sensor height.
	 */
	Translation(uint cols, uint rows);


	/**
	 * Creates index matrices of corresponding taxel positions in convolution matrix.
	 * @param cols Tactile sensor width.
	 * @param rows Tactile sensor height.
	 */
	void init(uint cols, uint rows);


	/**
	 * (Re)sets the reference tactile sensor matrix.
	 * Computes the reference frames's convolution with itself.
	 * @param referenceFrame The reference tactile sensor matrix.
	 */
	void setReferenceFrame(cv::Mat& referenceFrame);

	/**
	 * Computes the slip vector between the current and the reference tactile sensor matrix. (Normalized Cross Correlation)
	 * Use this method in conjunction with setReferenceFrame().
	 * @param currentFrame The current tactile sensor matrix.
	 * @return The corresponding slip vector.
	 */
	cv::Point2d computeSlipReference(cv::Mat& currentFrame);

	/**
	 * Computes the slip vector between the current and the previous tactile sensor matrix. (Normalized Cross Correlation)
	 * Use this method in conjunction with setReferenceFrame().
	 * @param currentFrame The current tactile sensor matrix.
	 * @return The corresponding slip vector.
	 */
	cv::Point2d computeSlip(cv::Mat& currentFrame);

};

/**
 * @class Rotation
 * @brief Implements the rotational slip detection based on the principal axis method.
 */
class Rotation {

private:

	double m_threshEccentricity; /// Ratio of principal axis lengths (disc or square: 0.0, elongated rectangle: ->1.0)
	double m_threshCompactness; /// How much the object resembles a disc or circle

	bool m_validReferenceAngle;
	bool m_validPreviousAngle;

	cv::Mat m_normalizedFrame;
	cv::Mat m_paddedFrame;

	int m_n; /// Full turn carry
	double m_referenceAngle; /// [0, 180), has to be initialized
	double m_previousAngle; /// [0, 360)
	double m_currentAngle; /// [0, 360)
	double m_slipAngleReference; /// Difference between reference and current angle
	double m_slipAnglePrevious; /// Difference between previous and current angle


	/**
	 * Computes signed difference between two angles (ignoring circular identities)
	 * @param angle0 The first angle.
	 * @param angle1 The second angle.
	 * @return
	 */
	double angleDifference(double angle0, double angle1);

public:

	/**
	 * Constructor.
	 */
	Rotation();


	/**
	 * Initialization / reseting of angle tracking
	 * @param referenceFrame The reference tactile sensor matrix.
	 * @return Success.
	 */
	bool setReferenceFrame(cv::Mat& referenceFrame);


	/**
	 * Computes the shape's orientation using the principal axis method.
	 * Shape features such as eccentricity and compactness can be used for quality evaluation.
	 * @param frame The current tactile sensor matrix.
	 * @return A tuple containing shape features and orientation.
	 */
	shapeFeatures rotationFromMoments(cv::Mat& frame);


	/**
	 * Computes slip angles by evaluating the orientation using rotationFromMoments() and tracking the rotation.
	 * @param currentFrame The current tactile sensor matrix.
	 * @return A tuple containing shape features, orientation and slip angles.
	 */
	rotationResult computeRotation(cv::Mat& currentFrame);

};


/**
 * @class SlipDetector
 * @brief Combined Slip-Detection class (Translation + Rotation)
 */
class SlipDetector {

private:
	boost::shared_ptr<Translation> translation;
	boost::shared_ptr<Rotation> rotation;

	bool m_translationInitialized;
	bool m_rotationInitialized;

	int m_referenceActiveCellsTranslation;
	int m_referenceActiveCellsRotation;

	int m_currentActiveCells;
	int m_thresholdActiveCellsTranslation;
	int m_thresholdActiveCellsRotation;

public:

	/**
	 * Constructor.
	 * It calls the translational and rotational slip-detection constructors.
	 * @param cols Tactile sensor width.
	 * @param rows Tactile sensor height.
	 */
	SlipDetector(uint cols, uint rows);


	/**
	 * Invalidates the reference frame, the previous frame as well as the tracked angle.
	 * @return void
	 */
	void reset();


	/**
	 * (Re)sets the reference frame for both translational and rotational slip-detection.
	 * Checks the number of active taxels
	 * @param referenceFrame The reference tactile sensor matrix.
	 * @return Success.
	 */
	bool setReferenceFrame(cv::Mat& referenceFrame);


	/**
	 * (Re)sets the reference frame for the translational slip-detection. Counts the number of active taxels.
	 * Translation fails only if frames are empty.
	 * @param referenceFrame The reference tactile sensor matrix.
	 * @return Success.
	 */
	bool setReferenceFrameTranslation(cv::Mat& referenceFrame);


	/**
	 * (Re)sets the reference frame for the translational slip-detection. Expects the number of active taxels.
	 * Translation fails only if frames are empty.
	 * @param referenceFrame The reference tactile sensor matrix.
	 * @param activeCells Number of active taxels.
	 * @return Success.
	 */
	bool setReferenceFrameTranslation(cv::Mat& referenceFrame, int activeCells);


	/**
	 * (Re)sets the reference frame for the rotational slip-detection. Counts the number of active taxels.
	 * Rotation fails if frame is empty and/or shape is circular.
	 * @param referenceFrame The reference tactile sensor matrix.
	 * @return Success.
	 */
	bool setReferenceFrameRotation(cv::Mat& referenceFrame);


	/**
	 * (Re)sets the reference frame for the rotational slip-detection. Expects the number of active taxels.
	 * Rotation fails if frame is empty and/or shape is circular.
	 * @param referenceFrame The reference tactile sensor matrix.
	 * @return Success.
	 */
	bool setReferenceFrameRotation(cv::Mat& referenceFrame, int activeCells);


	/**
	 * Performs both translational and rotational slip-detection.
	 * It is not necessary to set the reference or previous tactile sensor matrix beforehand.
	 * In this case, the methods are initialized with the current frame and the actual slip vector/rotation angle is computed between the very same tactile image.
	 * The real slip-detection then starts with the next call to this function, assuming the tactile sensor matrix satisfies the constraints.
	 * @param currentFrame The current tactile sensor matrix.
	 * @return The combined results of the rotational and translational slip-detection.
	 */
	slipResult computeSlip(cv::Mat& currentFrame);
};


#endif /* SLIPDETECTION_H_ */
