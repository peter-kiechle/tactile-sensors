#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "featureExtraction.h"
#include "Miniball.hpp"
#include "forwardKinematics.h"
#include "utils.h"

FeatureExtraction::FeatureExtraction(FrameManager& fm) : frameManager(fm) { }


FeatureExtraction::~FeatureExtraction() { }


std::vector<double> FeatureExtraction::computeCentroid(int frameID, int matrixID) {
	matrixInfo &matrixInfo = frameManager.getMatrixInfo(matrixID);
	double sum_x = 0.0;
	double sum_y = 0.0;
	double sum_intensity = 0.0;
	for(uint y = 0; y < matrixInfo.cells_y; y++) {
		for(uint x = 0; x < matrixInfo.cells_x; x++) {
			double value = static_cast<double>(frameManager.getFilteredTexel(frameID, matrixID, x, y) );
			if(value > 0.0) { // active cell
				sum_x += x*value;
				sum_y += y*value;
				sum_intensity += value;
			}
		}
	}
	std::vector<double> centroid(2, 0.0);
	if(sum_intensity > 0.0) {
		centroid[0] = sum_x / sum_intensity;
		centroid[1] = sum_y / sum_intensity;
	}
	return centroid;
}


array_type FeatureExtraction::computeMoments(int frameID, int matrixID, int pmax) {
	TSFrame* tsFrame = frameManager.getFilteredFrame(frameID);
	matrixInfo &matrixInfo = frameManager.getMatrixInfo(matrixID);
	cv::Mat frame = cv::Mat( matrixInfo.cells_y, matrixInfo.cells_x, CV_32F, tsFrame->cells.data()+matrixInfo.texel_offset).clone(); // Copying

	// Threshold to binary image
	//int thresh = 0;
	//cv::threshold(frame, frame, thresh, 4096.0, cv::THRESH_BINARY);

	// Scale such that highest intensity is 1.0
	//cv::normalize(frame, frame, 0.0, 1.0, NORM_MINMAX, CV_32F);

	// Scale [0..1] assuming max value is 4096 (12 bits, but I have never actually observed that value)
	frame.convertTo(frame, CV_32F, 1.0/4096.0);

	// Scale [0..1] using empirical max values
	//if(matrixID == 1) {
	//   frame.convertTo(frame, CV_32F, 1.0/3554.0);
	//}
	//else if(matrixID == 5) {
	//    frame.convertTo(frame, CV_32F, 1.0/2493.0);
	//}

	array_type T_pq_doubleprime(boost::extents[pmax][pmax]);
	if(activeCells(frameID, matrixID) > 0) {
		ChebyshevMoments.computeInvariants(frame, pmax, T_pq_doubleprime);
	} else {
		for(int p = 0; p < pmax; p++) {
			for(int q = 0; q < pmax; q++) {
				T_pq_doubleprime[p][q] = 0.0; // All moments are zero in an ampty frame
			}
		}
	}
	return T_pq_doubleprime;
}


double FeatureExtraction::computeStandardDeviation(int frameID, int matrixID) {
	matrixInfo &matrixInfo = frameManager.getMatrixInfo(matrixID);

	// Compute mean of active cells
	int nCells = 0;
	double sum = 0.0;
	for(uint y = 0; y < matrixInfo.cells_y; y++) {
		for(uint x = 0; x < matrixInfo.cells_x; x++) {
			double value = frameManager.getFilteredTexel(frameID, matrixID, x, y) / 4096.0;
			if(value > 0.0) { // active cell
				sum += value;
				nCells++;
			}
		}
	}

	if(nCells == 0) {
		return 0.0;
	}

	double matrixMean = sum / static_cast<double>(nCells);

	// Compute standard deviation of active cells
	sum = 0.0;
	for(uint y = 0; y < matrixInfo.cells_y; y++) {
		for(uint x = 0; x < matrixInfo.cells_x; x++) {
			double value = frameManager.getFilteredTexel(frameID, matrixID, x, y) / 4096.0;
			if(value > 0.0) { // active cell
				sum += (matrixMean-value)*(matrixMean-value);
			}
		}
	}
	double matrixStdDev = sqrt(sum / static_cast<double>(nCells));

	return matrixStdDev;
}


std::vector<double> FeatureExtraction::computeMiniball(int frameID, double phi0, double phi1, double phi2, double phi3, double phi4, double phi5, double phi6) {

	// Joint angles [phi0 .. phi6]
	std::vector<double> all_angles(7);
	all_angles[0] = utils::degToRad(phi0); // Rotational axis (Finger 0 + 2)
	all_angles[1] = utils::degToRad(phi1); // Finger 0
	all_angles[2] = utils::degToRad(phi2);
	all_angles[3] = utils::degToRad(phi3); // Finger 1
	all_angles[4] = utils::degToRad(phi4);
	all_angles[5] = utils::degToRad(phi5); // Finger 2
	all_angles[6] = utils::degToRad(phi6);

	ForwardKinematics forwardKinematics;
	forwardKinematics.setAngles(all_angles);

	// Prepare 3D data points of active cells
	std::vector<std::vector<double> > points;
	for(uint m = 0; m < frameManager.getNumMatrices(); m++) {
		matrixInfo &matrixInfo = frameManager.getMatrixInfo(m);
		int activeCells = 0;
		for(uint y = 0; y < matrixInfo.cells_y; y++) {
			for(uint x = 0; x < matrixInfo.cells_x; x++) {
				if(frameManager.getFilteredTexel(frameID, m, x, y) > 0.0) { // active cell
					std::vector<double> point = forwardKinematics.GetTaxelXYZ(m, x, y);
					points.push_back(point);
					activeCells++;
				}
			}
		}
	}

	// Compute the minimal bounding sphere of active cells
	typedef std::vector<std::vector<double> >::const_iterator PointIterator;
	typedef std::vector<double>::const_iterator CoordIterator;
	typedef Miniball::Miniball <Miniball::CoordAccessor<PointIterator, CoordIterator> > MB;
	int d = 3; // Dimension of bounding sphere
	MB mb(d, points.begin(), points.end());

	std::vector<double> result(4);

	// Center of the Miniball
	const double* center = mb.center();
	for(int i=0; i<d; ++i, ++center) {
		result[i] = *center;
	}

	// Radius
	result[3] = sqrt(mb.squared_radius());

	return result;
}


std::vector<double> FeatureExtraction::computeMiniball(int frameID, std::vector<double>& angles) {
	return computeMiniball(frameID, angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], angles[6]);
}


std::vector<double> FeatureExtraction::computeMiniballCentroid(int frameID, double phi0, double phi1, double phi2, double phi3, double phi4, double phi5, double phi6) {

	// Joint angles [phi0 .. phi6]
	std::vector<double> all_angles(7);
	all_angles[0] = utils::degToRad(phi0); // Rotational axis (Finger 0 + 2)
	all_angles[1] = utils::degToRad(phi1); // Finger 0
	all_angles[2] = utils::degToRad(phi2);
	all_angles[3] = utils::degToRad(phi3); // Finger 1
	all_angles[4] = utils::degToRad(phi4);
	all_angles[5] = utils::degToRad(phi5); // Finger 2
	all_angles[6] = utils::degToRad(phi6);

	ForwardKinematics forwardKinematics;
	forwardKinematics.setAngles(all_angles);

	// Compute center of mass of each active sensor matrix and determine corresponding 3D-coordinate on sensor plane
	std::vector<std::vector<double> > points;
	for(uint m = 0; m < frameManager.getNumMatrices(); m++) {
		matrixInfo &matrixInfo = frameManager.getMatrixInfo(m);
		int activeCells = 0;
		double sum_x = 0.0;
		double sum_y = 0.0;
		double sum_intensity = 0.0;
		for(uint y = 0; y < matrixInfo.cells_y; y++) {
			for(uint x = 0; x < matrixInfo.cells_x; x++) {
				double value = static_cast<double>(frameManager.getFilteredTexel(frameID, m, x, y) );
				if(value > 0.0) { // active cell
					activeCells++;
				sum_x += x*value;
				sum_y += y*value;
				sum_intensity += value;
				}
			}
		}
		if(activeCells > 0) {
			// Note: coordinates on matrix surface in mm (not cell index): (0,0) is center of top left cell
			double centroid_x = (sum_x / sum_intensity) * 3.4;
			double centroid_y = (sum_y / sum_intensity) * 3.4;
			std::vector<double> point = forwardKinematics.GetPointOnSensorPlaneXYZ(m, centroid_x, centroid_y);
			points.push_back(point);
		}
	}

	// Compute the minimal bounding sphere of active cells
	typedef std::vector<std::vector<double> >::const_iterator PointIterator;
	typedef std::vector<double>::const_iterator CoordIterator;
	typedef Miniball::Miniball <Miniball::CoordAccessor<PointIterator, CoordIterator> > MB;
	int d = 3; // Dimension of bounding sphere
	MB mb(d, points.begin(), points.end());

	std::vector<double> result(4);

	// Center of the MB
	const double* center = mb.center();
	for(int i=0; i<d; ++i, ++center) {
		result[i] = *center;
	}

	// Radius
	result[3] = sqrt(mb.squared_radius());

	return result;
}

std::vector<double> FeatureExtraction::computeMiniballCentroid(int frameID, std::vector<double>& angles) {
	return computeMiniballCentroid(frameID, angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], angles[6]);
}


std::vector<double> FeatureExtraction::computeMiniballPoints(std::vector< std::vector<double> >& taxels, double phi0, double phi1, double phi2, double phi3, double phi4, double phi5, double phi6) {

	// Joint angles [phi0 .. phi6]
	std::vector<double> all_angles(7);
	all_angles[0] = utils::degToRad(phi0); // Rotational axis (Finger 0 + 2)
	all_angles[1] = utils::degToRad(phi1); // Finger 0
	all_angles[2] = utils::degToRad(phi2);
	all_angles[3] = utils::degToRad(phi3); // Finger 1
	all_angles[4] = utils::degToRad(phi4);
	all_angles[5] = utils::degToRad(phi5); // Finger 2
	all_angles[6] = utils::degToRad(phi6);

	ForwardKinematics forwardKinematics;
	forwardKinematics.setAngles(all_angles);

	// Determine corresponding 3D-coordinates of points on sensor plane
	std::vector<std::vector<double> > points;
	for(uint i = 0; i < taxels.size(); i++) {
		std::vector<double> point = forwardKinematics.GetPointOnSensorPlaneXYZ(static_cast<int>(taxels[i][0]), taxels[i][1]*3.4, taxels[i][2]*3.4);
		points.push_back(point);
	}

	// Compute the minimal bounding sphere of active cells
	typedef std::vector<std::vector<double> >::const_iterator PointIterator;
	typedef std::vector<double>::const_iterator CoordIterator;
	typedef Miniball::Miniball <Miniball::CoordAccessor<PointIterator, CoordIterator> > MB;
	int d = 3; // Dimension of bounding sphere
	MB mb(d, points.begin(), points.end());

	std::vector<double> result(4);

	// Center of the MB
	const double* center = mb.center();
	for(int i=0; i<d; ++i, ++center) {
		result[i] = *center;
	}

	// Radius
	result[3] = sqrt(mb.squared_radius());

	return result;
}

std::vector<double> FeatureExtraction::computeMiniballPoints(std::vector< std::vector<double> >& taxels, std::vector<double>& angles) {
	return computeMiniballPoints(taxels, angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], angles[6]);
}
