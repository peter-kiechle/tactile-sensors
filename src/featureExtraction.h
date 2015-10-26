#ifndef FEATUREEXTRACTION_H_
#define FEATUREEXTRACTION_H_

#include "framemanager.h"
#include "chebyshevMoments.h"

/**
 * @class FeatureExtraction
 * @brief Unifies the computation of features, in detail: The standard deviation, Chebyshev moments and minimal bounding sphere.
 */
class FeatureExtraction {

private:
	FrameManager& frameManager;
	Chebyshev ChebyshevMoments;
	inline int activeCells(int frameID, int matrixID) { return frameManager.getFrameProcessor()->getMatrixNumActiveCells(frameID, matrixID); };

public:

	/**
	 * Constructor
	 */
	FeatureExtraction(FrameManager& fm);
	virtual ~FeatureExtraction();


	/**
	 * Computes center of mass (in texel coordinates)
	 * @param frameID The frameID.
	 * @param matrixID The matrixID.
	 * @return The centroid.
	 */
	std::vector<double> computeCentroid(int frameID, int matrixID);


	/**
	 * Computes Chebyshev moments
	 * @param frameID The frameID.
	 * @param matrixID The matrixID.
	 * @param pmax Maximum moment order.
	 * @return The Chebyshev moments.
	 */
	array_type computeMoments(int frameID, int matrixID, int pmax);


	/**
	 * Computes standard deviation of tactile sensor frames (intensity values, not 2D image moments). Only active cells are considered
	 * @param frameID The frameID.
	 * @param matrixID The matrixID.
	 * @return The standard deviation.
	 */
	double computeStandardDeviation(int frameID, int matrixID);


	/**
	 * Compute the minimal bounding sphere of active cells. See overloaded variant.
	 * @param frameID The frameID.
	 * @param phi0,phi1,phi2,phi3,phi4,phi5,phi6 Joint angles [phi0 .. phi6].
	 * @return Center and radius of miniball.
	 */
	std::vector<double> computeMiniball(int frameID, double phi0, double phi1, double phi2, double phi3, double phi4, double phi5, double phi6);


	/**
	 * Compute the minimal bounding sphere of active cells, overloaded variant.
	 * @param frameID The frameID.
	 * @param angles Joint angles [phi0 .. phi6].
	 * @return Center and radius of miniball.
	 */
	std::vector<double> computeMiniball(int frameID, std::vector<double>& angles);


	/**
	 * Compute the minimal bounding sphere based on the per matrix centroid of active cells. See overloaded variant.
	 * @param frameID The frameID.
	 * @param phi0,phi1,phi2,phi3,phi4,phi5,phi6 Joint angles [phi0 .. phi6].
	 * @return Center and radius of miniball.
	 */
	std::vector<double> computeMiniballCentroid(int frameID, double phi0, double phi1, double phi2, double phi3, double phi4, double phi5, double phi6);


	/**
	 * Compute the minimal bounding sphere based on the per matrix centroid of active cells, overloaded variant.
	 * @param frameID The frameID.
	 * @param angles Joint angles [phi0 .. phi6].
	 * @return Center and radius of miniball.
	 */
	std::vector<double> computeMiniballCentroid(int frameID, std::vector<double>& angles);


	/**
	 * Compute the minimal bounding sphere based on the specified taxels. See overloaded variant.
	 * @param taxels Vector of specified taxels per matrix .
	 * @param angles Joint angles [phi0 .. phi6].
	 * @return Center and radius of miniball.
	 */
	std::vector<double> computeMiniballPoints(std::vector< std::vector<double> >& taxels, double phi0, double phi1, double phi2, double phi3, double phi4, double phi5, double phi6);


	/**
	 * Compute the minimal bounding sphere based on the specified taxels, overloaded variant.
	 * @param taxels Vector of specified taxels per matrix .
	 * @param angles Joint angles [phi0 .. phi6].
	 * @return Center and radius of miniball.
	 */
	std::vector<double> computeMiniballPoints(std::vector< std::vector<double> >& taxels, std::vector<double>& angles);
};

#endif /* FEATUREEXTRACTION_H_ */
