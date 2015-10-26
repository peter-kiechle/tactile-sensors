#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "slipdetection.h"


/*****************************************************************
 * Translation
 *****************************************************************/

Translation::Translation(uint cols, uint rows) {
	init(cols, rows);
}


void Translation::init(uint cols, uint rows) {

	m_cols = cols;
	m_rows = rows;
	m_cols_C = 2*m_cols-1;
	m_rows_C = 2*m_rows-1;

	// A (repeat rows)
	m_A = cv::Mat::ones(0, m_cols_C, CV_32F); // Zero rows
	cv::Mat row = cv::Mat::ones(1, m_cols_C, CV_32F);
	int zeroingOffset = (m_cols_C+1)/2;
	for(int i = 0; i < m_cols_C; i++) {
		row.at<float>(i) = (i+1)-zeroingOffset;
	}
	cv::repeat(row, m_cols_C, 1, m_A); // Repeat rows

	// B (repeat columns)
	m_B = cv::Mat::ones(m_rows_C, 0, CV_32F); // Zero columns
	cv::Mat col = cv::Mat::ones(m_rows_C, 1, CV_32F);
	zeroingOffset = (m_rows_C+1)/2;
	for(int i = 0; i < m_rows_C; i++) {
		col.at<float>(i) = (i+1)-zeroingOffset;
	}
	cv::repeat(col, 1, m_rows_C, m_B); // Repeat columns
}


cv::Mat Translation::flip(cv::Mat M) {
	cv::Mat result;
	cv::flip(M, result, -1); // -1 = both axes
	return result;
}


void Translation::convolve2d(cv::Mat &referenceFrame, cv::Mat& currentFrame, cv::Mat& C) {
	cv::Mat referenceFrame_padded;

	// Zero padding like scipy's convolve2d: mode='full', boundary='fill', fillvalue=0
	// Note: The padding is asymmetric since frame size (and hence the kernel) is even,
	int top = m_rows/2;
	int bottom = (m_rows-1)/2;
	int left = m_cols/2;
	int right = (m_cols-1)/2;

	cv::copyMakeBorder(referenceFrame, referenceFrame_padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0) );
	cv::Point anchor(m_cols-m_cols/2 - 1, m_rows-m_rows/2 - 1); // Note: symmetric kernel...
	cv::filter2D(referenceFrame_padded, C, referenceFrame_padded.depth(), flip(currentFrame), anchor, 0, cv::BORDER_CONSTANT);
}


void Translation::setReferenceFrame(cv::Mat& referenceFrame) {
	cv::Mat m_C_reference;
	cv::Mat means_columns, means_columns_T;
	cv::Mat means_rows, means_rows_T;

	referenceFrame.copyTo(m_referenceFrame); // Deep copy of reference frame
	m_previousFrame = referenceFrame;

	// Convolution of reference frame with itself
	convolve2d(referenceFrame, referenceFrame, m_C_reference);

	// Means of columns (reduce along y-axis)
	cv::reduce(m_C_reference, means_columns, 0, CV_REDUCE_AVG, CV_32F); // Reduced to a single row
	cv::transpose(means_columns, means_columns_T);

	// Means of rows (reduce along x-axis)
	cv::reduce(m_C_reference, means_rows, 1, CV_REDUCE_AVG, CV_32F); // Reduced to a single column
	cv::transpose(means_rows, means_rows_T);

	cv::Mat shift_columns = (m_A * means_columns_T) / cv::sum(means_columns)[0];
	float shift_x = cv::mean(shift_columns)[0];

	cv::Mat shift_rows = (means_rows_T * m_B) / cv::sum(means_rows)[0];
	float shift_y = cv::mean(shift_rows)[0];

	m_referenceDisplacement = cv::Point2d(shift_x, shift_y);
	m_previousDisplacement = m_referenceDisplacement;
}


cv::Point2d Translation::computeSlipReference(cv::Mat& currentFrame) {
	cv::Mat C_current;
	cv::Mat means_columns, means_columns_T;
	cv::Mat means_rows, means_rows_T;

	// Convolution of reference frame with comparison frame
	convolve2d(m_referenceFrame, currentFrame, C_current);

	// Means of columns (reduce along y-axis)
	cv::reduce(C_current, means_columns, 0, CV_REDUCE_AVG, CV_32F); // Reduced to a single row
	cv::transpose(means_columns, means_columns_T);

	// Means of rows (reduce along x-axis)
	cv::reduce(C_current, means_rows, 1, CV_REDUCE_AVG, CV_32F); // Reduced to a single column
	cv::transpose(means_rows, means_rows_T);

	cv::Mat shift_columns = (m_A * means_columns_T) / cv::sum(means_columns)[0];
	float shift_x = cv::mean(shift_columns)[0];

	cv::Mat  shift_rows = (means_rows_T * m_B) / cv::sum(means_rows)[0];
	float shift_y = cv::mean(shift_rows)[0];

	cv::Point2d currentDisplacement(shift_x, shift_y);

	cv::Point2d slipvector = currentDisplacement - m_referenceDisplacement;

	return slipvector;
}


cv::Point2d Translation::computeSlip(cv::Mat& currentFrame) {
	cv::Mat C_current;
	cv::Mat means_columns, means_columns_T;
	cv::Mat means_rows, means_rows_T;

	// Convolution of reference frame with comparison frame
	convolve2d(m_previousFrame, currentFrame, C_current);

	// Means of columns (reduce along y-axis)
	cv::reduce(C_current, means_columns, 0, CV_REDUCE_AVG, CV_32F); // Reduced to a single row
	cv::transpose(means_columns, means_columns_T);

	// Means of rows (reduce along x-axis)
	cv::reduce(C_current, means_rows, 1, CV_REDUCE_AVG, CV_32F); // Reduced to a single column
	cv::transpose(means_rows, means_rows_T);

	cv::Mat shift_columns = (m_A * means_columns_T) / cv::sum(means_columns)[0];
	float shift_x = cv::mean(shift_columns)[0];

	cv::Mat  shift_rows = (means_rows_T * m_B) / cv::sum(means_rows)[0];
	float shift_y = cv::mean(shift_rows)[0];

	cv::Point2d currentDisplacement(shift_x, shift_y);

	cv::Point2d slipvector = currentDisplacement - m_previousDisplacement;

	m_previousFrame = currentFrame;
	m_previousDisplacement = currentDisplacement;

	return slipvector;
}



/*****************************************************************
 * Rotation
 *****************************************************************/

Rotation::Rotation() {
	m_threshEccentricity = 0.6; // Ratio of principal axis lengths (disc or square: 0.0, elongated rectangle: ->1.0)
	// m_threshEccentricity = 0.1; // Another definition: Percental difference of eigenvalues (ellipse axis lengths)
	m_threshCompactness = 0.9; // How much the object resembles a disc (perfect circle: 1.0)
}


bool Rotation::setReferenceFrame(cv::Mat& referenceFrame) {

	bool success;
	cv::Point2d centroid, skew;
	double angle, lambda1, lambda2, eccentricity, compactness;

	// Compute orientation
	boost::tie(success, centroid, skew,	angle, lambda1, lambda2, eccentricity, compactness) = rotationFromMoments(referenceFrame);

	if(success) {
		m_validReferenceAngle = true;
		m_validPreviousAngle = true;
		m_referenceAngle = angle;
		m_previousAngle = angle;
	} else {
		m_validReferenceAngle = false;
		m_validPreviousAngle = false;
	}
	m_n = 0; // Full turn carry

	return success;
}


double Rotation::angleDifference(double angle0, double angle1) {
	double difference = angle1 - angle0;

	while (difference < -180.0) {
		difference += 360;
	}

	while (difference > 180.0) {
		difference -= 360;
	}

	return difference;
}


shapeFeatures Rotation::rotationFromMoments(cv::Mat& frame) {

	// Normalize frame such that M['m00'] = 1
	// Only necessary for computation of standard deviation and skew
	frame.convertTo(m_normalizedFrame, CV_64F);
	m_normalizedFrame = m_normalizedFrame / cv::sum(m_normalizedFrame)[0]; // Sum of intensity values

	// Compute image moments using OpenCV
	cv::Moments M = cv::moments(m_normalizedFrame, false); // binary image = false

	assert(M.m00 > 0.0 && "Moments not defined for empty frame!");

	// Normalized moments
	double mu11_, mu02_, mu20_;
	mu11_ = M.mu11/M.m00;
	mu02_ = M.mu02/M.m00;
	mu20_ = M.mu20/M.m00;

	// Eigenvalues
	double lambda1 = 0.5*(mu20_ + mu02_) + 0.5 * sqrt(mu20_*mu20_ + mu02_*mu02_ - 2*mu20_*mu02_ + 4*mu11_*mu11_);
	double lambda2 = 0.5*(mu20_ + mu02_) - 0.5 * sqrt(mu20_*mu20_ + mu02_*mu02_ - 2*mu20_*mu02_ + 4*mu11_*mu11_);

	// Geometric moment based statistics:
	// From "Moments and Moment Invariants in Pattern Recognition":
	// In case of zero means, m20 and m02 are variances of horizontal
	// and vertical projections and m11 is a covariance between them.

	// Center of mass
	cv::Point2d centroid;
	centroid.x = M.m10 / M.m00;
	centroid.y = M.m01 / M.m00;

	// 2D Standard deviation
	//cv::Point2d variance, stdDev;
	//variance.x = m20 - centroid.x * M.m10;
	//variance.y = m02 - centroid.y * M.m01;
	//stdDev.x = sqrt(variance.x);
	//stdDev.y = sqrt(variance.y);

	// Skew
	cv::Point2d skew;
	if(fabs(M.mu20) > 0.001) {
		skew.x = M.mu30 / pow(M.mu20, 1.5);
	} else {
		skew.x = 0.0;
	}
	if(fabs(M.mu02) > 0.001) {
		skew.y = M.mu03 / pow(M.mu02, 1.5);
	} else {
		skew.y = 0.0;
	}

	// The principal axis method produces the best results if the ellipse of inertia is elongated
	// In the degenerated case (ellipse -> circle) the method becomes numerically unstable, but may still be applicable (if skewness != 0).
	// In any case, there is no actual orientation if the profile's shape resembles a disc. (periodic symmetry < pi)

	// Compute eccentricity measure
	double eccentricity = sqrt( 1.0 - (lambda2/lambda1)*(lambda2/lambda1) ); //  Ratio of principal axis lengths (disc or square: 0.0, elongated rectangle: ->1.0)
	//double eccentricity = fabs(lambda1 - lambda2) / (0.5*(lambda1 + lambda2)); // Another definition: Percental eigenvalue difference

	// Compute compactness measure
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	// Add border padding due to strange behavior of cv::findContours() in case of contours touching the image border
	cv::copyMakeBorder(frame, m_paddedFrame, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0.0, 0.0, 0.0) );
	cv::normalize(m_paddedFrame, m_paddedFrame, 0, 255, NORM_MINMAX, CV_8UC1); // Scale such that highest intensity is 255
	cv::findContours(m_paddedFrame, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	double perimeter = cv::arcLength(contours[0], true);
	double area = cv::contourArea(contours[0]);
	double compactness = (4.0 * M_PI * area) / (perimeter*perimeter);

	bool success = true;
	if(eccentricity < m_threshEccentricity && compactness > m_threshCompactness) {
		success = false; // Unfortunately we are dealing with a circular object
	}

	// Compute orientation angle
	double orientation = 0.0;
	if(success) {
		// ±90 degrees (in relation to positive x-axis)
		double angle_rad = (0.5*atan2((2*M.mu11), (M.mu20-M.mu02)));
		orientation = angle_rad * 180.0/M_PI;

		// Clamp to [0, 180)
		if(orientation < 0) {
			orientation = 90.0 + (90.0 + orientation);
		}
	}
	return boost::tie(success, centroid, skew, orientation, lambda1, lambda2, eccentricity, compactness);
}


rotationResult Rotation::computeRotation(cv::Mat& currentFrame) {
	bool success;
	cv::Point2d centroid, skew;
	double orientation, lambda1, lambda2, eccentricity, compactness;

	// Compute orientation
	boost::tie(success, centroid, skew,	orientation, lambda1, lambda2, eccentricity, compactness) = rotationFromMoments(currentFrame);

	// Track rotation
	if(success) {
		if(m_validPreviousAngle) {
			// Track slip angle between current orientation [0, 180) and previous angle [0, 360)
			double currentAngle1 = orientation;
			double currentAngle2 = 180.0 + orientation;

			double angleDiff;
			double angleDiff1 = angleDifference(m_previousAngle, currentAngle1);
			double angleDiff2 = angleDifference(m_previousAngle, currentAngle2);

			if( fabs(angleDiff1) < fabs(angleDiff2) ) {
				m_currentAngle = currentAngle1;
				angleDiff = angleDiff1;
			} else {
				m_currentAngle = currentAngle2;
				angleDiff = angleDiff2;
			}

			// Discontinuity 360 -> 0
			if (m_currentAngle < m_previousAngle && angleDiff > 0.0) {
				m_n += 1;
			}

			// Discontinuity 0 -> 360
			if (m_currentAngle > m_previousAngle && angleDiff < 0.0) {
				m_n -= 1;
			}

			// Difference to reference angle
			if(m_validReferenceAngle) {
				m_slipAngleReference = m_n*360.0 + m_currentAngle + angleDifference(m_referenceAngle, 0.0);
			}

			// Difference to previous angle
			m_slipAnglePrevious = angleDiff;

		} else { // Computation of last frame's orientation failed, hence no slip
			m_slipAnglePrevious = 0.0;
		}

		m_previousAngle = m_currentAngle;
		m_validPreviousAngle = true;

	} else { // Principal axis method failed
		m_validPreviousAngle = false;
		m_slipAnglePrevious = 0.0;
	}

	return boost::tie(success, m_slipAnglePrevious, m_slipAngleReference, orientation, centroid, skew, lambda1, lambda2, eccentricity, compactness);
}


/*****************************************************************
 * Translation + Rotation
 *****************************************************************/

SlipDetector::SlipDetector(uint cols, uint rows) {
	translation = boost::make_shared<Translation>(cols, rows);
	rotation = boost::make_shared<Rotation>();
	m_translationInitialized = false;
	m_rotationInitialized = false;
	m_thresholdActiveCellsTranslation = 1;
	m_thresholdActiveCellsRotation = 2;
}


void SlipDetector::reset() {
	m_translationInitialized = false;
	m_rotationInitialized = false;
}


bool SlipDetector::setReferenceFrameTranslation(cv::Mat& referenceFrame) {
	return setReferenceFrameTranslation(referenceFrame, cv::countNonZero(referenceFrame));
}


bool SlipDetector::setReferenceFrameTranslation(cv::Mat& referenceFrame, int activeCells) {
	m_translationInitialized = false;
	m_referenceActiveCellsTranslation = activeCells;

	// Check if minimum amount of sensor cells are active
	if(m_referenceActiveCellsTranslation >= m_thresholdActiveCellsTranslation) {
		translation->setReferenceFrame(referenceFrame); // Always works if at least one cell is active
		m_translationInitialized = true;
	}
	return m_translationInitialized;
}


bool SlipDetector::setReferenceFrameRotation(cv::Mat& referenceFrame) {
	return setReferenceFrameRotation(referenceFrame, cv::countNonZero(referenceFrame));
}


bool SlipDetector::setReferenceFrameRotation(cv::Mat& referenceFrame, int activeCells) {
	m_rotationInitialized = false;
	m_referenceActiveCellsRotation = activeCells;

	// Check if minimum amount of sensor cells are active
	if(m_referenceActiveCellsRotation >= m_thresholdActiveCellsRotation) {
		m_rotationInitialized = rotation->setReferenceFrame(referenceFrame); // Additional constraints have to be met
	}
	return m_rotationInitialized;
}


bool SlipDetector::setReferenceFrame(cv::Mat& referenceFrame) {
	int activeCells = cv::countNonZero(referenceFrame);
	setReferenceFrameTranslation(referenceFrame, activeCells);
	setReferenceFrameRotation(referenceFrame, activeCells);

	return m_translationInitialized && m_rotationInitialized;
}


slipResult SlipDetector::computeSlip(cv::Mat& currentFrame) {

	// Ensure that a reference frame has been set before further processing
	m_currentActiveCells =  cv::countNonZero(currentFrame);
	if(!m_translationInitialized) {
		setReferenceFrameTranslation(currentFrame, m_currentActiveCells);
	}
	if(!m_rotationInitialized) {
		setReferenceFrameRotation(currentFrame, m_currentActiveCells);
	}

	// Compute slip vector
	bool successTranslation;
	cv::Point2d slipVector, slipVectorReference;
	if(m_translationInitialized && m_currentActiveCells > m_thresholdActiveCellsTranslation) {
		successTranslation = true;
		slipVector = translation->computeSlip(currentFrame);
		slipVectorReference = translation->computeSlipReference(currentFrame);
	} else {
		successTranslation = false;
		slipVector = cv::Point2d(0.0, 0.0);
		slipVectorReference = cv::Point2d(0.0, 0.0);
	}

	// Compute slip angle
	bool successRotation;
	double slipAngle, slipAngleReference, orientation, lambda1, lambda2, eccentricity, compactness;
	cv::Point2d centroid, skew;
	if(m_rotationInitialized && m_currentActiveCells > m_thresholdActiveCellsRotation) {
		boost::tie(successRotation, slipAngle, slipAngleReference, orientation,
				centroid, skew, lambda1, lambda2, eccentricity, compactness) = rotation->computeRotation(currentFrame);
	} else {
		successRotation = false;
		slipAngle = 0.0;
		slipAngleReference = 0.0;
		orientation = 0.0;
		centroid = cv::Point2d(0.5*currentFrame.cols, 0.5*currentFrame.rows);
		skew = cv::Point2d(0.0, 0.0);
		lambda1 = 1.0;
		lambda2 = 1.0;
		eccentricity = 0.0;
		compactness = 1.0;
	}

	// Combine results of slip vector and slip angle computation
	slipResult result = { successTranslation,
			successRotation,
			slipVector.x,
			slipVector.y,
			slipVectorReference.x,
			slipVectorReference.y,
			slipAngle,
			slipAngleReference,
			orientation,
			centroid.x,
			centroid.y,
			skew.x,
			skew.y,
			lambda1,
			lambda2,
			eccentricity,
			compactness
	};

	return result;
}
