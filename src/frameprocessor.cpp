
#include "framemanager.h"
#include "frameprocessor.h"

FrameProcessor::FrameProcessor() {
	m_filter = FILTER_NONE;
	m_currentFrameID = -1;
}

FrameProcessor::~FrameProcessor() { }


void FrameProcessor::setFrameManager(FrameManager * fm) {
	frameManager = fm;
}


int FrameProcessor::getNumActiveCells(uint frameID) {
	int numActiveCells = 0;
	for(uint m = 0; m < frameManager->getNumMatrices(); m++) {
		numActiveCells += getMatrixNumActiveCells(frameID, m);
	}
	return numActiveCells;
}


int FrameProcessor::getMatrixNumActiveCells(uint frameID, uint matrixID) {
	//TSFrame* tsFrame = frameManager->getFrame(frameID);
	TSFrame* tsFrame = frameManager->getFilteredFrame(frameID);
	matrixInfo &matrixInfo = frameManager->getMatrixInfo(matrixID);
	SensorMatrixMap matrixMap(tsFrame->cells.data()+matrixInfo.texel_offset, matrixInfo.cells_y, matrixInfo.cells_x);
	int numActiveCells = 0;

	for(int y = 0; y < matrixMap.rows(); y++){
		for(int x = 0; x < matrixMap.cols(); x++){
			if(matrixMap(y,x) > 0) {
				numActiveCells++;
			}
		}
	}
	return numActiveCells;
}


void FrameProcessor::calcCharacteristics(uint frameID) {
	m_matrixAverage.clear();
	m_matrixMin.clear();
	m_matrixMax.clear();

	//TSFrame* tsFrame = frameManager->getFrame(frameID);
	TSFrame* tsFrame = frameManager->getFilteredFrame(frameID);

	// Compute matrix characteristics
	for(uint m = 0; m < frameManager->getNumMatrices(); m++) {
		matrixInfo &matrixInfo = frameManager->getMatrixInfo(m);
		SensorMatrixMap matrixMap(tsFrame->cells.data()+matrixInfo.texel_offset, matrixInfo.cells_y, matrixInfo.cells_x);

		int activeCells = 0;
		double sum = 0.0;
		double min = 9999.0; // 12 bit values only
		double max = 0.0;
		for(int y = 0; y < matrixMap.rows(); y++){
			for(int x = 0; x < matrixMap.cols(); x++){
				if(matrixMap(y,x) > 0) { // Only consider active cells
					sum += matrixMap(y,x);
					activeCells++;
					if(matrixMap(y,x) > max) {
						max = matrixMap(y,x);
					}
					if(matrixMap(y,x) < min) {
						min = matrixMap(y,x);
					}
				}
			}
		}
		if(activeCells > 0) {
			//m_matrixAverage.push_back( sum / static_cast<double>(matrixInfo.num_cells) );
			m_matrixAverage.push_back( sum / static_cast<double>(activeCells) );
			m_matrixMin.push_back(min);
			m_matrixMax.push_back(max);
		} else {
			m_matrixAverage.push_back(0.0);
			m_matrixMin.push_back(0.0);
			m_matrixMax.push_back(0.0);
		}
	}

	// Compute frame characteristics
	int activeMatrices = 0;
	double sum = 0.0;
	double min = 4095.0;
	double max = 0.0;
	for(uint i = 0; i < m_matrixAverage.size(); i++) {
		if(m_matrixAverage[i] > 0) { // Only consider matrices with active cells
			sum += m_matrixAverage[i];
			activeMatrices++;
			if(m_matrixMax[i] > max) {
				max = m_matrixMax[i];
			}
			if(m_matrixMin[i] < min) {
				min = m_matrixMin[i];
			}
		}
	}
	if(activeMatrices > 0) {
		//m_frameAverage = sum / static_cast<double>(frameManager->getNumCells());
		m_frameAverage = sum / static_cast<double>(activeMatrices);
		m_frameMin = min;
		m_frameMax = max;
	} else {
		m_frameAverage = 0.0;
		m_frameMin = 0.0;
		m_frameMax = 0.0;
	}
}


double FrameProcessor::getAverage(uint frameID) {
	if(m_currentFrameID != static_cast<int>(frameID)) { // Perform calculations only once
		calcCharacteristics(frameID);
		m_currentFrameID = frameID;
	}
	return m_frameAverage;
}


double FrameProcessor::getMatrixAverage(uint frameID, uint matrixID) {
	if(m_currentFrameID != static_cast<int>(frameID)) { // Perform calculations only once
		calcCharacteristics(frameID);
		m_currentFrameID = frameID;
	}
	return m_matrixAverage[matrixID];
}


double FrameProcessor::getMin(uint frameID) {
	if(m_currentFrameID != static_cast<int>(frameID)) { // Perform calculations only once
		calcCharacteristics(frameID);
		m_currentFrameID = frameID;
	}
	return m_frameMin;
}


double FrameProcessor::getMatrixMin(uint frameID, uint matrixID) {
	if(m_currentFrameID != static_cast<int>(frameID)) { // Perform calculations only once
		calcCharacteristics(frameID);
		m_currentFrameID = frameID;
	}
	return m_matrixMin[matrixID];
}


double FrameProcessor::getMax(uint frameID) {
	if(m_currentFrameID != static_cast<int>(frameID)) { // Perform calculations only once
		calcCharacteristics(frameID);
		m_currentFrameID = frameID;
	}
	return m_frameMax;
}


double FrameProcessor::getMatrixMax(uint frameID, uint matrixID) {
	if(m_currentFrameID != static_cast<int>(frameID)) { // Perform calculations only once
		calcCharacteristics(frameID);
		m_currentFrameID = frameID;
	}
	return m_matrixMax[matrixID];
}


void FrameProcessor::setFilterNone() {
	m_filter = FILTER_NONE;
}


void FrameProcessor::setFilterMedian(int kernelRadius, bool masked) {
	m_filter = FILTER_MEDIAN;
	m_kernelRadius = kernelRadius;
	m_masked = masked;
}


void FrameProcessor::setFilterMedian3D(bool masked) {
	m_filter = FILTER_MEDIAN3D;
	m_masked = masked;
}


void FrameProcessor::setFilterGaussian(int kernelRadius, double sigma, int borderType) {
	m_filter = FILTER_GAUSSIAN;
	m_kernelRadius = kernelRadius;
	m_sigma = sigma;
}


void FrameProcessor::setFilterBilateral(int kernelRadius, double sigmaColor, double sigmaSpace, int borderType) {
	m_filter = FILTER_BILATERAL;
	m_kernelRadius = kernelRadius;
	m_sigmaColor = sigmaColor;
	m_sigmaSpace = sigmaSpace;
	m_borderType = borderType;
}


void FrameProcessor::setFilterOpening(int kernelType, int kernelRadius, bool masked, int borderType) {
	m_filter = FILTER_OPENING;
	m_kernelRadius = kernelRadius;
	m_masked = masked;
	m_borderType = borderType;

	// Create kernel
	int d = 2*m_kernelRadius+1; // Kernel size guaranteed to be odd

	switch(kernelType) {
	case 0: m_kernelType = cv::MORPH_CROSS; break;
	case 1: m_kernelType = cv::MORPH_RECT; break;
	case 2: m_kernelType = cv::MORPH_ELLIPSE; break;
	default: m_kernelType = cv::MORPH_CROSS; break;
	}

	m_kernelMorphological = cv::getStructuringElement(m_kernelType, cv::Size(d, d), cv::Point( -1, -1 ) );
}


FilterType FrameProcessor::getFilterType() {
	return m_filter;
}


double FrameProcessor::calcGaussianSigma(int kernelRadius) {
	double sigma = (kernelRadius+1) / 2.35482004503; // Full Width at Half Maximum (FWHM) of k+1 pixel ( Factor: 2*sqrt(2*ln(2)) )
	return sigma;
}


void FrameProcessor::applyFilter(TSFrame* tsFrame, int frameID) {

	if(frameID >= 0) { // negative frameIDs indicate live frames
		// Create a copy of the requested frame
		std::vector<float>::iterator from = frameManager->getFrame(frameID)->cells.begin();
		std::vector<float>::iterator to   = frameManager->getFrame(frameID)->cells.end();
		std::copy(from, to, tsFrame->cells.begin());
	}

	int d = 2*m_kernelRadius+1; // Kernel size guaranteed to be odd

	for(uint m = 0; m < frameManager->getNumMatrices(); m++) {
		matrixInfo &matrixInfo = frameManager->getMatrixInfo(m);
		cv::Mat matrix = cv::Mat(matrixInfo.cells_y, matrixInfo.cells_x, CV_32F, tsFrame->cells.data()+matrixInfo.texel_offset); // No copying

		if(m_filter == FILTER_MEDIAN) {
			if(m_masked) {
				cv::Mat mask;
				cv::medianBlur(matrix, mask, d);

				// Create a binary mask of the medianBlur result and keep the original nonzero values of the sensor frame
				cv::threshold(mask, mask, 0.0, 1.0, cv::THRESH_BINARY_INV);
				mask.convertTo(mask, CV_8U);
				cv::Mat zeros = cv::Mat::zeros(matrix.size[0], matrix.size[1], CV_32F);
				zeros.copyTo(matrix, mask); // Overwrite unmasked elements of filter with zeros
			} else {
				cv::medianBlur(matrix, matrix, d);
			}
		}

		else if(m_filter == FILTER_MEDIAN3D) {

			// Just some dummy matrix for masking
			cv::Mat mask;
			cv::Mat original;
			if(m_masked) {
				mask.create(matrix.size[0], matrix.size[1], CV_32F);
				original = matrix;
				matrix = mask;
			}

			// Hard-coded 3x3x3 median filter
			int radius_depth = 1;
			int radius_width = 1;
			int radius_height = 1;
			int kernel_depth = 2*radius_depth+1;
			int kernel_width = 2*radius_width+1;
			int kernel_height = 2*radius_height+1;

			std::vector<float> kernel(kernel_depth * kernel_width * kernel_height);
			std::vector<cv::Mat> matrices;

			// Create padded version of relevant matrices
			for(int depth = 0; depth < kernel_depth; depth++) {
				TSFrame* currentTsFrame;
				if(frameID-depth < 0) { // Not enough frames available back in time -> repeat first frame
					currentTsFrame = frameManager->getFrame(frameID);
				} else {
					currentTsFrame = frameManager->getFrame(frameID-depth);
				}
				cv::Mat currentMatrix = cv::Mat(matrixInfo.cells_y, matrixInfo.cells_x, CV_32F, currentTsFrame->cells.data()+matrixInfo.texel_offset); // No copying
				cv::Mat currentMatrix_padded;
				cv::copyMakeBorder(currentMatrix, currentMatrix_padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);
				matrices.push_back(currentMatrix_padded);
			}

			// Iterate through image space
			for (uint x_outer = 1; x_outer < matrixInfo.cells_x+1; x_outer++) {
				for (uint y_outer = 1; y_outer < matrixInfo.cells_y+1; y_outer++) {


					/*
					// 3D Median
					// Collect kernel elements centered around (d, x, y)
					int k = 0;
					for(int depth = 0; depth < kernel_depth; depth++) {
						for(uint x_inner = x_outer - radius_width; x_inner <= x_outer + radius_width; x_inner++) {
							for(uint y_inner = y_outer - radius_height; y_inner <= y_outer + radius_height; y_inner++) {
								kernel[k++] = matrices[depth].at<float>(y_inner, x_inner);
							}
						}
					}
					// Find median
					std::vector<float>::iterator first = kernel.begin();
					std::vector<float>::iterator last = kernel.end();
					std::vector<float>::iterator middle = first + (last - first) / 2;
					std::nth_element(first, middle, last);
					matrix.at<float>(y_outer-1, x_outer-1) = *middle;
					 */

					// Temporal Minimum + 2D Median
					for(uint x_inner = x_outer - radius_width; x_inner <= x_outer + radius_width; x_inner++) {
						for(uint y_inner = y_outer - radius_height; y_inner <= y_outer + radius_height; y_inner++) {
							float min = 9999;
							for(int depth = 0; depth < kernel_depth; depth++) {
								float val = matrices[depth].at<float>(y_inner, x_inner);
								if(val < min) {
									min = val;
								}
							}
							//matrix.at<float>(y_outer-1, x_outer-1) = min;
							matrices[0].at<float>(y_inner, x_inner) = min;
							matrices[1].at<float>(y_inner, x_inner) = min;
							matrices[2].at<float>(y_inner, x_inner) = min;
						}
					}

					// Collect kernel elements centered around (d, x, y)
					int k = 0;
					for(int depth = 0; depth < kernel_depth; depth++) {
						for(uint x_inner = x_outer - radius_width; x_inner <= x_outer + radius_width; x_inner++) {
							for(uint y_inner = y_outer - radius_height; y_inner <= y_outer + radius_height; y_inner++) {
								kernel[k++] = matrices[depth].at<float>(y_inner, x_inner);
							}
						}
					}
					// Find median
					std::vector<float>::iterator first = kernel.begin();
					std::vector<float>::iterator last = kernel.end();
					std::vector<float>::iterator middle = first + (last - first) / 2;
					std::nth_element(first, middle, last);
					matrix.at<float>(y_outer-1, x_outer-1) = *middle;
				}
			}

			if(m_masked) {
				// Create a binary mask of opening result and keep the original nonzero values of the sensor matrix
				cv::threshold(matrix, matrix, 0.0, 1.0, cv::THRESH_BINARY_INV);
				matrix.convertTo(matrix, CV_8U);
				cv::Mat zeros = cv::Mat::zeros(matrix.size[0], matrix.size[1], CV_32F);
				zeros.copyTo(original, matrix); // Overwrite unmasked elements of filter with zeros

			}
		}

		else if(m_filter == FILTER_GAUSSIAN) {
			cv::GaussianBlur(matrix, matrix, cv::Size(d, d), m_sigma, 0, m_borderType );
		}

		else if(m_filter == FILTER_BILATERAL) {
			cv::Mat copy = matrix.clone(); // bilateralFilter() does not work in-place
			cv::bilateralFilter(copy, matrix, d, m_sigmaColor, m_sigmaSpace, m_borderType );
		}

		else if(m_filter == FILTER_OPENING) {
			if(m_masked) {
				cv::Mat mask;
				cv::morphologyEx(matrix, mask, cv::MORPH_OPEN, m_kernelMorphological, cv::Point(-1,-1), 1, m_borderType);

				// Create a binary mask of opening result and keep the original nonzero values of the sensor matrix
				cv::threshold(mask, mask, 0.0, 1.0, cv::THRESH_BINARY_INV);
				mask.convertTo(mask, CV_8U);
				cv::Mat zeros = cv::Mat::zeros(matrix.size[0], matrix.size[1], CV_32F);
				zeros.copyTo(matrix, mask); // Overwrite unmasked elements of filter with zeros
			} else {
				cv::morphologyEx(matrix, matrix, cv::MORPH_OPEN, m_kernelMorphological, cv::Point(-1,-1), 1, m_borderType);
			}
		}
	}
}
