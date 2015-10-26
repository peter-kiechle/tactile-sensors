
#ifndef PREPROCESSOR_H_
#define PREPROCESSOR_H_

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/// Copy of TSFrame cells
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SensorMatrix;

/// Mapping to underlying data in TSFrame cells
typedef Eigen::Map<SensorMatrix> SensorMatrixMap;

/// Filter name
enum FilterType { FILTER_NONE, FILTER_MEDIAN, FILTER_MEDIAN3D, FILTER_GAUSSIAN, FILTER_BILATERAL, FILTER_OPENING };

class FrameManager;
struct TSFrame;

/**
 * @class FrameProcessor
 * @brief Manages temporal, spatial and spatio-temporal filtering
 */
class FrameProcessor {

private:
	FrameManager *frameManager;
	int m_currentFrameID;

	// Characteristics of single matrices
	std::vector<double> m_matrixAverage;
	std::vector<double> m_matrixMin;
	std::vector<double> m_matrixMax;

	// Combined matrix characteristics
	double m_frameAverage;
	double m_frameMin;
	double m_frameMax;

	// 2D Filtering
	FilterType m_filter;
	int m_borderType;
	int m_kernelType;
	int m_kernelRadius;
	bool m_masked;
	double m_sigma;
	double m_sigmaColor;
	double m_sigmaSpace;
	cv::Mat m_kernelMorphological;

public:

	/**
	 * Constructor.
	 */
	FrameProcessor();

	virtual ~FrameProcessor();

	/**
	 * Sets the frame manager.
	 * @param fm The frame manager.
	 * @return void
	 */
	void setFrameManager(FrameManager * fm);


	/**
	 * Returns the number of active taxels of the entire frame
	 * @param frameID The frame ID.
	 * @return The number of active taxels.
	 */
	int getNumActiveCells(uint frameID);


	/**
	 * Returns the number of active taxels of the specified matrix.
	 * @param frameID The frame ID.
	 * @param matrixID The matrix ID.
	 * @return The number of active taxels.
	 */
	int getMatrixNumActiveCells(uint frameID, uint matrixID);


	/**
	 * Calculates all characteristic values at once in a single iteration.
	 * Characteristic values are: Per matrix as well as per frame averages, minimum and maximum values
	 * @param frameID The frame ID.
	 * @return void
	 */
	void calcCharacteristics(uint frameID);


	/**
	 * Returns frame average. See calcCharacteristics().
	 * @param frameID The frame ID.
	 * @return void
	 */
	double getAverage(uint frameID);


	/**
	 * Returns the matrix average. See calcCharacteristics().
	 * @param frameID The frame ID.
	 * @param matrixID The matrix ID.
	 * @return void
	 */
	double getMatrixAverage(uint frameID, uint matrixID);


	/**
	 * Returns frame minimum value. See calcCharacteristics().
	 * @param frameID The frame ID.
	 * @return void
	 */
	double getMin(uint frameID);


	/**
	 * Returns the matrix minimum value. See calcCharacteristics().
	 * @param frameID The frame ID.
	 * @param matrixID The matrix ID.
	 * @return void
	 */
	double getMatrixMin(uint frameID, uint matrixID);


	/**
	 * Returns frame maximum value. See calcCharacteristics().
	 * @param frameID The frame ID.
	 * @return void
	 */
	double getMax(uint frameID);


	/**
	 * Returns the matrix maximum value. See calcCharacteristics().
	 * @param frameID The frame ID.
	 * @param matrixID The matrix ID.
	 * @return void
	 */
	double getMatrixMax(uint frameID, uint matrixID);


	/**
	 * Disables filtering.
	 */
	void setFilterNone();


	/**
	 * Enables the 2D Median filter.
	 * @param kernelRadius The filtering kernel' radius.
	 * @param masked Taxels that survived the filtering process retain their original values.
	 * @return void
	 */
	void setFilterMedian(int kernelRadius, bool masked);


	/**
	 * Enables the spatio-temporal 3x3x3 Median filter.
	 * @param masked Taxels that survived the filtering process retain their original values.
	 * @return void
	 */
	void setFilterMedian3D(bool masked);


	/**
	 * Enables the Gaussian filter.
	 * @param kernelRadius The filtering kernel' radius.
	 * @param sigma The standard deviation.
	 * @param borderType OpenCV border type.
	 * @return void
	 */
	void setFilterGaussian(int kernelRadius, double sigma, int borderType);


	/**
	 * Enables the Bilateral filter.
	 * @param kernelRadius The filtering kernel' radius.
	 * @param sigmaColor The color standard deviation parameter.
	 * @param sigmaSpace The spatial standard deviation parameter.
	 * @param borderType OpenCV border type.
	 * @return void
	 */
	void setFilterBilateral(int kernelRadius, double sigmaColor, double sigmaSpace, int borderType);


	/**
	 * Enables the opening operation.
	 * @param kernelType OpenCV kernel type.
	 * @param kernelRadius The filtering kernel' radius.
	 * @param masked Taxels that survived the filtering process retain their original values.
	 * @param borderType OpenCV border type.
	 * @return void
	 */
	void setFilterOpening(int kernelType, int kernelRadius, bool masked, int borderType);


	/**
	 * Getter: filter type.
	 * @return The filter type.
	 */
	FilterType getFilterType();

	double calcGaussianSigma(int kernelRadius);

	/**
	 * Performs the actual filtering based on the current filter settings.
	 * @param tsFrame Pointer to the tactile sensor frame.
	 * @param frameID The frame ID.
	 * @return void
	 */
	void applyFilter(TSFrame* tsFrame, int frameID);

};

#endif /* PREPROCESSOR_H_ */
