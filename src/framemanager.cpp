#include <algorithm>
#include <stdint.h>

// *.dsa3 files are bzip2 encoded
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/bzip2.hpp>

// XML parsing
#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "framemanager.h"
#include "framegrabberSDH.h"
#include "framegrabberDSA.h"

#include "utils.h"

using namespace boost::iostreams;
using namespace boost::property_tree;


FrameManager::FrameManager() {
	frameProcessor.setFrameManager(this);
	resetOffline();
}

FrameManager::~FrameManager() { }


void FrameManager::resetOffline() {
	hand = NULL;
	ts = NULL;
	frameGrabberSDH = NULL;
	frameGrabberDSA = NULL;

	m_liveFrame.reset();

	m_sensor.nb_matrices = 0;
	m_sensor.nb_cells = 0;
	m_sensor.generated_by = 0;
	m_sensor.hw_revision = 0;
	m_sensor.serial_no = 0;
	m_sensor.converter_resolution = 0;

	m_matrices.clear();
	m_tsframes.clear();
	m_temperatureFrames.clear();
	m_jointAngleFrames.clear();

	m_slipDetectors.clear();
	m_liveSlipDetection = false;
	m_slipDetectorState.resize(6,false);
	m_liveSlip.resize(6, boost::optional<slipResult> () );
	m_liveSlipAvailable = false;
	m_threshSlipReference = 1.0;
	m_threshSlipConsecutive = 0.2;

	m_tsFrameAvailable = false;
	m_jointAngleFrameAvailable = false;
	m_currentFrameID = 0;
	m_profileName = "No Profile available";
}


void FrameManager::resetOnline() {

	m_tsframes.clear();
	m_temperatureFrames.clear();
	m_jointAngleFrames.clear();

	m_tsFrameAvailable = false;
	m_jointAngleFrameAvailable = false;
	m_currentFrameID = 0;

	m_slipResultQueue.clear();
	for(uint m = 0; m < 6; m++) {
		m_slipDetectors[m]->reset();
		m_slipResultQueue.push_back( boost::make_shared<std::queue<slipResult> > () );
	}

	// Mark uninitialized frame with a distinctive pattern (useful for debugging)
	for(uint m = 0; m < getNumMatrices(); m++) {
		for(uint y = 0; y < m_matrices[m].cells_y; y++) {
			for(uint x = 0; x < m_matrices[m].cells_x; x++) {

				if(m % 2 == 0) { // Proximal
					if( x == 0 || x == m_matrices[m].cells_x-1 || y == 0 || y == m_matrices[m].cells_y-1 ) {
						uint cellID = m_matrices[m].texel_offset + y * m_matrices[m].cells_x + x;
						m_liveFrame->cells[cellID] = 1000.0;
					}
				} else { // Distal
					if( x == 0 || x == m_matrices[m].cells_x-1 || y == 0 || y == m_matrices[m].cells_y-1
							|| (y < 6 && (x == 1 || x == m_matrices[m].cells_x-2)) ) {
						uint cellID = m_matrices[m].texel_offset + y * m_matrices[m].cells_x + x;
						m_liveFrame->cells[cellID] = 1000.0;
					}
				}
			}
		}
	}
	liveJointAngleFrame.angles = std::vector<double> (7, 0.0);
	liveJointAngleFrame.timestamp = 0;
	m_profileName = "Live";
}


void FrameManager::setSDH(cSDH *sdh) {
	hand = sdh;
}


void FrameManager::setDSA(cDSA *dsa) {
	ts = dsa;
	queryDSAInfo();

	m_selectedCells.assign(getNumCells(), false); // Reset selected cells

	m_liveFrame = boost::shared_ptr<TSFrame>(new TSFrame(m_sensor.nb_cells));

	m_slipResultQueue.clear();
	for(uint m = 0; m < 6; m++) {
		m_slipResultQueue.push_back( boost::make_shared<std::queue<slipResult> > () );
	}

	initFilter();
}


bool FrameManager::isConnectedSDH() {
	if(hand) {
		return hand->IsOpen();
	} else {
		return false;
	}
}

bool FrameManager::isConnectedDSA() {
	if(ts) {
		return ts->IsOpen();
	} else {
		return false;
	}
}


void FrameManager::queryDSAInfo() {
	// Query sensor controller info
	m_sensor.nb_matrices = ts->GetSensorInfo().nb_matrices;
	m_sensor.generated_by = ts->GetSensorInfo().generated_by;
	m_sensor.hw_revision = ts->GetSensorInfo().hw_revision;
	m_sensor.serial_no = ts->GetSensorInfo().serial_no;
	m_sensor.converter_resolution = 12; // NOTE: Will always be 12 Bit...
	m_sensor.nb_cells = 0;

	// Query matrix info
	m_matrices.clear();
	m_matrixSensitivity.resize(m_sensor.nb_matrices, 1.0);
	m_matrixThreshold.resize(m_sensor.nb_matrices, 1.0);
	m_slipDetectors.clear();
	for(uint m = 0; m < m_sensor.nb_matrices; m++) {
		matrixInfo matrix;
		const cDSA::sMatrixInfo& matrixInfo = ts->GetMatrixInfo(m);

		std::copy(&matrixInfo.uid[0], &matrixInfo.uid[6], matrix.uid);
		matrix.hw_revision = matrixInfo.hw_revision;
		matrix.cells_x = matrixInfo.cells_x;
		matrix.cells_y = matrixInfo.cells_y;
		matrix.texel_width = matrixInfo.texel_width;
		matrix.texel_height = matrixInfo.texel_width;
		matrix.matrix_center_x = matrixInfo.matrix_center_x;
		matrix.matrix_center_y = matrixInfo.matrix_center_y;
		matrix.matrix_center_z = matrixInfo.matrix_center_z;
		matrix.matrix_theta_x = matrixInfo.matrix_theta_x;
		matrix.matrix_theta_y = matrixInfo.matrix_theta_y;
		matrix.matrix_theta_z = matrixInfo.matrix_theta_z;
		matrix.fullscale = (1 << m_sensor.converter_resolution)-1;

		// Query matrix sensitivity/threshold
		cDSA::sSensitivityInfo sensitivity_info = ts->GetMatrixSensitivity(m);
		m_matrixSensitivity[m] = sensitivity_info.cur_sens;
		m_matrixThreshold[m] = ts->GetMatrixThreshold(m);

		// Convert matrix masks from string to bit vector
		std::string staticMaskString = ts->GetMatrixMaskString(cDSA::MASK_STATIC, m);
		for(std::string::size_type i = 0; i < staticMaskString.size(); ++i) {
			staticMaskString[i] == '1' ? matrix.static_mask.push_back(true) : matrix.static_mask.push_back(false);
		}
		std::string dynamicMaskString = ts->GetMatrixMaskString(cDSA::MASK_DYNAMIC, m);
		for(std::string::size_type i = 0; i < dynamicMaskString.size(); ++i) {
			dynamicMaskString[i] == '1' ? matrix.dynamic_mask.push_back(true) : matrix.dynamic_mask.push_back(false);
		}

		matrix.num_cells = matrix.cells_x * matrix.cells_y;
		matrix.texel_offset = m_sensor.nb_cells;

		m_matrices.push_back(matrix);
		m_sensor.nb_cells += matrix.num_cells;

		// Init slip detectors
		m_slipDetectors.push_back( boost::make_shared<SlipDetector>(matrix.cells_x, matrix.cells_y) );
	}
}


void FrameManager::setSensitivity(uint matrixID, float sensitivity) {
	try
	{
		if (sensitivity < 0.0 || sensitivity > 1.0) {
			return;
		}

		UInt8 sensitivityBit = 1; // matrix support for changing sensitivity
		if( (ts->GetMatrixInfo(matrixID).feature_flags >> sensitivityBit) & 1) { // Check for support
			printf("Changing Sensitivity of Matrix %d to %.2f ...\n", matrixID, sensitivity);
			ts->SetMatrixSensitivity(matrixID, sensitivity, false, false, false );
		} else {
			printf("Changing Sensitivity of Matrix %d not possible!\n", matrixID);  // Matrix 0 does not support this)
		}

	} catch ( cSDHLibraryException* e )
	{
		std::cerr << "\nSet matrix sensitivity: Caught exception from SDHLibrary: " << e->what() << ". Giving up!\n";
		delete e;
	}
	catch (...)
	{
		std::cerr << "\nCaught unknown exception, giving up\n";
	}

	// Query matrix sensitivity
	cDSA::sSensitivityInfo sensitivity_info = ts->GetMatrixSensitivity(matrixID);
	m_matrixSensitivity[matrixID] = sensitivity_info.cur_sens;
}


void FrameManager::setThreshold(uint matrixID, float threshold) {
	try
	{
		if (threshold < 0.0 || threshold > 4095) {
			return;
		}
		printf("Changing Threshold of Matrix %d to %.2f ...\n", matrixID, threshold);
		ts->SetMatrixThreshold(matrixID, threshold, false, false, false );

	} catch (cSDHLibraryException* e)	{
		std::cerr << "\nSet matrix threshold: Caught exception from SDHLibrary: " << e->what() << ". Giving up!\n";
		delete e;
	}
	catch (...)	{
		std::cerr << "\nCaught unknown exception, giving up\n";
	}
	// Query matrix threhold
	m_matrixThreshold[matrixID] = ts->GetMatrixThreshold(matrixID);
}

void FrameManager::setFrameGrabberDSA(FrameGrabberDSA *fgDSA) {
	frameGrabberDSA = fgDSA;
}


void FrameManager::setFrameGrabberSDH(FrameGrabberSDH *fgSDH) {
	frameGrabberSDH = fgSDH;
}


uint FrameManager::getCurrentFrameID() {
	return m_currentFrameID;
}


void FrameManager::setCurrentFrameID(uint frameID) {
	m_currentFrameID = frameID;
}


bool FrameManager::getTSFrameAvailable() {
	return m_tsFrameAvailable;
}


void FrameManager::setTSFrameAvailable(bool value) {
	m_tsFrameAvailable = value;
}


bool FrameManager::getJointAngleFrameAvailable() {
	return m_jointAngleFrameAvailable;
}


void FrameManager::setJointAngleFrameAvailable(bool value) {
	m_jointAngleFrameAvailable = value;
}


void FrameManager::setLiveFrame() {
	for(uint i = 0; i < m_sensor.nb_cells; i++) {
		m_liveFrame->cells[i] = static_cast<float>(ts->GetFrame().texel[i]);
	}
	m_tsFrameAvailable = true;

	if(m_liveSlipDetection) {
		slipResultProducer(); // Compute slip and put result in queue
	}
}


TSFrame& FrameManager::allocateTSFrame() {
	m_tsframes.push_back(TSFrame(m_sensor.nb_cells));
	return m_tsframes.back();
}


void FrameManager::addTSFrame() {

	// Allocate space for new tactile sensor frame
	m_tsframes.push_back(TSFrame(m_sensor.nb_cells));

	// Store volatile sensor data (also: convert uint16_t to float)
	for(uint i = 0; i < m_sensor.nb_cells; i++) {
		m_tsframes.back().cells[i] = ts->GetFrame().texel[i];
	}

	// Add time stamp
	m_tsframes.back().timestamp = utils::getCurrentTimeMilliseconds();
	m_tsFrameAvailable = true;
}


void FrameManager::deleteTSFrame(uint frameID) {
	m_tsframes.erase(m_tsframes.begin()+frameID); // expensive!!!

	// Update temperature mapping
	if(m_temperatureFrames.size() > 0) {
		m_temperatureMapping.erase(m_temperatureMapping.begin()+frameID); // expensive!!!
	}

	// Update joint angle mapping
	if(m_jointAngleFrames.size() > 0) {
		m_jointAngleMapping.erase(m_jointAngleMapping.begin()+frameID); // expensive!!!
	}

	if(m_currentFrameID > m_tsframes.size()-1) {
		m_currentFrameID--;
	}
}


void FrameManager::cropToFrames(uint64_t timestampFrom, uint64_t timestampTo) {

	if(m_tsframes.size() > 0) {
		// Find left/right indices of Tactile sensor frames
		std::deque<TSFrame>::iterator itLeftTS;
		std::deque<TSFrame>::iterator itRightTS;
		itLeftTS = std::lower_bound(m_tsframes.begin(), m_tsframes.end(), timestampFrom, TimestampComparator<TSFrame>() ); // Position right before timestampFrom
		itRightTS = std::upper_bound(m_tsframes.begin(), m_tsframes.end(), timestampTo, TimestampComparator<TSFrame>() ); // Position right after timestampTo
		m_tsframes.erase(itRightTS, m_tsframes.end()); // Delete tail [itRight, last)
		m_tsframes.erase(m_tsframes.begin(), itLeftTS); // Delete head [first, itLeft)
	}

	if(m_temperatureFrames.size() > 0) {
		// Find left/right indices of temperature frames
		std::deque<TemperatureFrame>::iterator itLeftTemp;
		std::deque<TemperatureFrame>::iterator itRightTemp;
		itLeftTemp = std::lower_bound(m_temperatureFrames.begin(), m_temperatureFrames.end(), timestampFrom, TimestampComparator<TemperatureFrame>() );
		itRightTemp = std::upper_bound(m_temperatureFrames.begin(), m_temperatureFrames.end(), timestampTo, TimestampComparator<TemperatureFrame>() );
		m_temperatureFrames.erase(itRightTemp, m_temperatureFrames.end()); // Delete tail [itRight, last)
		m_temperatureFrames.erase(m_temperatureFrames.begin(), itLeftTemp); // Delete head [first, itLeft)
	}

	if(m_jointAngleFrames.size() > 0) {
		// Find left/right indices of joint angle frames
		std::deque<JointAngleFrame>::iterator itLeftJointAngle;
		std::deque<JointAngleFrame>::iterator itRightJointAngle;
		itLeftJointAngle = std::lower_bound(m_jointAngleFrames.begin(), m_jointAngleFrames.end(), timestampFrom, TimestampComparator<JointAngleFrame>() );
		itRightJointAngle = std::upper_bound(m_jointAngleFrames.begin(), m_jointAngleFrames.end(), timestampTo, TimestampComparator<JointAngleFrame>() );
		m_jointAngleFrames.erase(itRightJointAngle, m_jointAngleFrames.end()); // Delete tail [itRright, last)
		m_jointAngleFrames.erase(m_jointAngleFrames.begin(), itLeftJointAngle); // Delete head [first, itLeft)
	}

	// Rebuild correspondence mappings
	createTemperatureMapping();
	createJointAngleMapping();

	if(m_currentFrameID > m_tsframes.size()-1) {
		m_currentFrameID = m_tsframes.size()-1;
	}
}


TSFrame* FrameManager::getFrame(uint frameID) {
	return &m_tsframes[frameID];
}


TSFrame* FrameManager::getCurrentFrame() {
	// Live mode
	if(frameGrabberDSA) {
		if(frameGrabberDSA->isCapturing()) {
			return m_liveFrame.get();
		}
	}
	// Not capturing or offline mode
	if(m_tsframes.size() == 0) { // No recorded frames available
		return m_liveFrame.get();
	} else {  // Return recorded frame
		return &m_tsframes[m_currentFrameID];
	}
}


TSFrame* FrameManager::getFilteredFrame(uint frameID) {
	if(frameProcessor.getFilterType() == FILTER_NONE) {
		return getFrame(frameID);
	} else {
		if(m_currentFilteredFrameID != frameID) { // Perform filtering only once
			frameProcessor.applyFilter(m_filteredFrame.get(), frameID);
			m_currentFilteredFrameID = frameID;
		}
		return m_filteredFrame.get();

	}
}


TSFrame* FrameManager::getCurrentFilteredFrame() {
	if(frameGrabberDSA) {
		if(frameGrabberDSA->isCapturing()) { // Live mode
			if(frameProcessor.getFilterType() == FILTER_NONE) {
				return m_liveFrame.get();
			} else {
				// TODO: Make access to liveFrame thread-safe
				// Create a copy of the requested frame
				std::vector<float>::iterator from = m_liveFrame.get()->cells.begin();
				std::vector<float>::iterator to   = m_liveFrame.get()->cells.end();
				std::copy(from, to, m_filteredFrame->cells.begin());

				// Apply filter
				frameProcessor.applyFilter(m_filteredFrame.get(), -1);
				return m_filteredFrame.get();
			}
		}
	}

	if(m_tsframes.size() == 0) { // No recorded frames available and not capturing
		if(frameProcessor.getFilterType() == FILTER_NONE) {
			return m_liveFrame.get();
		} else {
			frameProcessor.applyFilter(m_liveFrame.get(), -1);
			return m_liveFrame.get();
		}
	} else {  // Return recorded frame
		return getFilteredFrame(m_currentFrameID);
	}
}


float FrameManager::getTexel(uint frameID, uint m, uint x, uint y ) {
	return m_tsframes[frameID].cells[m_matrices[m].texel_offset + y * m_matrices[m].cells_x + x];
}


float FrameManager::getTexel(uint frameID, uint cellID) {
	return m_tsframes[frameID].cells[cellID];
}


float FrameManager::getFilteredTexel(uint frameID, uint m, uint x, uint y ) {
	if(frameProcessor.getFilterType() == FILTER_NONE) {
		return m_tsframes[frameID].cells[m_matrices[m].texel_offset + y * m_matrices[m].cells_x + x];
	} else {
		if(m_currentFilteredFrameID != frameID) { // Perform filtering only once
			frameProcessor.applyFilter(m_filteredFrame.get(), frameID);
			m_currentFilteredFrameID = frameID;
		}
		return m_filteredFrame->cells[m_matrices[m].texel_offset + y * m_matrices[m].cells_x + x];
	}
}


float FrameManager::getFilteredTexel(uint frameID, uint cellID) {
	if(frameProcessor.getFilterType() == FILTER_NONE) {
		return m_tsframes[frameID].cells[cellID];
	} else {
		if(m_currentFilteredFrameID != frameID) { // Perform filtering only once
			frameProcessor.applyFilter(m_filteredFrame.get(), frameID);
			m_currentFilteredFrameID = frameID;
		}
		return m_filteredFrame->cells[cellID];
	}
}


uint FrameManager::getFrameCountTS() {
	return m_tsframes.size();
}


uint FrameManager::getFrameCountTemperature() {
	return m_temperatureFrames.size();
}


uint FrameManager::getFrameCountJointAngles() {
	return m_jointAngleFrames.size();
}


void FrameManager::requestTemperatureFrame(bool record) {
	if(isConnectedSDH()) {
		// Temperatures 0-6: close to axes motors
		// Temperature 7: FPGA
		// Temperature 8: Printed circuit board
		TemperatureFrame temperatureFrame;
		temperatureFrame.values = hand->GetTemperature(hand->all_temperature_sensors);

		// Add time stamp in milliseconds
		temperatureFrame.timestamp = utils::getCurrentTimeMilliseconds();

		//printTemperatureFrame(temperatureFrame);

		if(record) {
			m_temperatureFrames.push_back(temperatureFrame);
		}
	} else {
		printf("Not adding temperature frame\n");
	}
}


TemperatureFrame* FrameManager::getTemperatureFrame(uint tempID) {
	return &m_temperatureFrames[tempID];
}


void FrameManager::requestJointAngleFrame(bool record) {

	// Get actual axis angle of all real axes
	// 0 : common base axis of finger 0 and 2
	// 1 : proximal axis of finger 0
	// 2 : distal axis of finger 0
	// 3 : proximal axis of finger 1
	// 4 : distal axis of finger 1
	// 5 : proximal axis of finger 2
	// 6 : distal axis of finger 2

	if(isConnectedSDH()) {
		liveJointAngleFrame.angles = hand->GetAxisActualAngle(hand->all_real_axes);

		// Add time stamp in milliseconds
		liveJointAngleFrame.timestamp = utils::getCurrentTimeMilliseconds();

		//printJointAngleFrame(jointAngleFrame);

		m_jointAngleFrameAvailable = true;

		if(record) {
			m_jointAngleFrames.push_back(liveJointAngleFrame);
		}
	} else {
		printf("Not adding joint angle frame\n");
	}
}


JointAngleFrame*  FrameManager::getJointAngleFrame(uint angleID) {
	return &m_jointAngleFrames[angleID];
}


JointAngleFrame* FrameManager::getCurrentJointAngleFrame() {
	// Live mode
	if(frameGrabberSDH) {
		if( frameGrabberSDH->isCapturing() ) {
			return &liveJointAngleFrame;
		}
	}

	// Not capturing or offline mode
	if(m_jointAngleFrames.size() == 0) { // No recorded frames available
		return &liveJointAngleFrame;
	} else {  // Return recorded frame
		return getCorrespondingJointAngle(m_currentFrameID);
	}
}


void FrameManager::createTemperatureMapping() {
	// Create mapping between tactile sensor frame and associated temperature
	uint nFrames = m_tsframes.size();
	uint nTemps = m_temperatureFrames.size();
	if(nTemps > 0) {
		m_temperatureMapping.resize(nFrames);
		uint tempID = 0;
		for(uint frameID = 0; frameID < nFrames; frameID++) {
			if(m_tsframes[frameID].timestamp <= m_temperatureFrames[0].timestamp) { // No temperature available yet
				m_temperatureMapping[frameID] = 0;
			} else {

				// Linear search for next time stamp (next comparison should usually be a hit)
				while(tempID < nTemps) {
					if(m_tsframes[frameID].timestamp < m_temperatureFrames[tempID+1].timestamp) {
						m_temperatureMapping[frameID] = tempID;
						break;
					}
					tempID++;
				}
				if(tempID > nTemps-1) { // No temperature available anymore
					m_temperatureMapping[frameID] = nTemps-1;
				}
			}
		}
	}
}


void FrameManager::createJointAngleMapping() {
	// Create mapping between tactile sensor frame and associated joint angles
	uint nAngles = m_jointAngleFrames.size();
	uint nFrames = m_tsframes.size();
	if(nAngles > 0) {
		m_jointAngleMapping.resize(nFrames);
		uint angleID = 0;
		for(uint frameID = 0; frameID < nFrames; frameID++) {
			if(m_tsframes[frameID].timestamp <= m_jointAngleFrames[0].timestamp) { // No joint angles available yet
				m_jointAngleMapping[frameID] = 0;
			} else {
				// Linear search for next time stamp (next comparison should usually be a hit)
				while(angleID < nAngles) {
					if(m_tsframes[frameID].timestamp < m_jointAngleFrames[angleID+1].timestamp) {
						m_jointAngleMapping[frameID] = angleID;
						break;
					}
					angleID++;
				}

				if(angleID > nAngles-1) { // No joint angles available anymore
					m_jointAngleMapping[frameID] = nAngles-1;
				}
			}
		}
	}
}


TemperatureFrame* FrameManager::getCorrespondingTemperature(uint frameID) {
	assert(m_temperatureMapping.size() > 0 && "No temperature mapping available");
	uint tempID = m_temperatureMapping[frameID];
	return getTemperatureFrame(tempID);
}


JointAngleFrame* FrameManager::getCorrespondingJointAngle(uint frameID) {
	assert(m_jointAngleMapping.size() > 0 && "No joint angle mapping available");
	uint angleID = m_jointAngleMapping[frameID];
	return getJointAngleFrame(angleID);
}


void FrameManager::selectCell(uint cellID, bool value) {
	m_selectedCells[cellID] = value;
}


bool FrameManager::isSelected(int cellID) {
	return m_selectedCells[cellID];
}


int FrameManager::getNumSelectedCells() {
	int numSelectedCells = 0;
	for(uint i = 0; i < m_selectedCells.size(); i++) {
		if(m_selectedCells[i] == true) {
			numSelectedCells++;
		}
	}
	return numSelectedCells;
}


std::vector<bool>& FrameManager::getSelection() {
	return m_selectedCells;
}


std::vector<int> FrameManager::createSelectedCellsIdx() {
	std::vector<int> selection;
	for(uint id = 0; id < m_selectedCells.size(); id++) {
		if(m_selectedCells[id]) {
			selection.push_back(id);
		}
	}
	return selection;
}


void FrameManager::setDynamicMask(std::vector<bool>& bitmask) {
	assert(bitmask.size() == getNumCells() && "Bitmask size differs from actual number of cells");

	// Pause grabbing for a moment to prevent conflicts on the serial port
	bool isGrabbing = false;
	if(frameGrabberDSA->isCapturing()) {
		isGrabbing = true;
		frameGrabberDSA->pause();
	}

	// Construct mask
	uint m, x, y;
	for(uint cellID = 0; cellID < getNumCells(); cellID++) {
		convertCellIndex(cellID, m, x, y);
		ts->SetMatrixMaskCell(m, x, y, bitmask[cellID]);
	}

	// Send mask to DSACON32m
	ts->SetMatrixMasks();

	// Update internal representation
	for(uint m = 0; m < m_sensor.nb_matrices; m++) {
		std::string dynamicMaskString = ts->GetMatrixMaskString(cDSA::MASK_DYNAMIC, m);
		m_matrices[m].dynamic_mask.clear();
		for(std::string::size_type i = 0; i < dynamicMaskString.size(); ++i) {
			dynamicMaskString[i] == '1' ? m_matrices[m].dynamic_mask.push_back(true) : m_matrices[m].dynamic_mask.push_back(false);
		}
	}

	// Resume grabbing
	if(isGrabbing) {
		frameGrabberDSA->resume();
	}
}


bool FrameManager::getStaticMask(uint m, uint x, uint y) {
	int index = y * m_matrices[m].cells_x + x ;
	return m_matrices[m].static_mask[index];
}


bool FrameManager::getDynamicMask(uint m, uint x, uint y) {
	int index = y * m_matrices[m].cells_x + x;
	return m_matrices[m].dynamic_mask[index];
}


FrameProcessor* FrameManager::getFrameProcessor() {
	return &frameProcessor;
}

void FrameManager::initFilter() {
	if(!m_filteredFrame) {
		m_filteredFrame = boost::shared_ptr<TSFrame>(new TSFrame(m_sensor.nb_cells));
	}
	m_currentFilteredFrameID = m_currentFrameID;
}


void FrameManager::setFilterNone() {
	frameProcessor.setFilterNone();
	// Invalidate filtered frame
	m_currentFilteredFrameID = -1; // Warning: Nothing is guaranteed here. But value is probably 4294967295
}


void FrameManager::setFilterMedian(int kernelRadius, bool masked) {
	frameProcessor.setFilterMedian(kernelRadius, masked);
	// Invalidate filtered frame
	m_currentFilteredFrameID = -1; // Warning: Nothing is guaranteed here. But value is probably 4294967295
}


void FrameManager::setFilterMedian3D(bool masked) {
	frameProcessor.setFilterMedian3D(masked);
	// Invalidate filtered frame
	m_currentFilteredFrameID = -1; // Warning: Nothing is guaranteed here. But value is probably 4294967295
}


void FrameManager::setFilterGaussian(int kernelRadius, double sigma, int borderType) {
	double s;
	if(sigma <= 0) { // Automatic determination of sigma
		s = frameProcessor.calcGaussianSigma(kernelRadius);
	} else {
		s = sigma;
	}
	frameProcessor.setFilterGaussian(kernelRadius, s, borderType);

	// Invalidate filtered frame
	m_currentFilteredFrameID = -1; // Warning: Nothing is guaranteed here. But value is probably 4294967295
}


void FrameManager::setFilterBilateral(int kernelSize, double sigmaColor, double sigmaSpace, int borderType) {
	frameProcessor.setFilterBilateral(kernelSize, sigmaColor, sigmaSpace, borderType);
	// Invalidate filtered frame
	m_currentFilteredFrameID = -1; // Warning: Nothing is guaranteed here. But value is probably 4294967295
}


void FrameManager::setFilterMorphological(int kernelType, int kernelRadius, bool masked, int borderType) {
	frameProcessor.setFilterOpening(kernelType, kernelRadius, masked, borderType);
	// Invalidate filtered frame
	m_currentFilteredFrameID = -1; // Warning: Nothing is guaranteed here. But value is probably 4294967295
}


boost::shared_ptr<SlipDetector> FrameManager::getSlipDetector(uint matrixID) {
	return m_slipDetectors[matrixID];
}


void FrameManager::enableSlipDetection(uint matrixID) {
	m_slipDetectorState[matrixID] = true;
	m_liveSlipDetection = true;
}


void FrameManager::disableSlipDetection(uint matrixID) {
	m_slipDetectorState[matrixID] = false;
	// Check if at least a single slip detector is still enabled
	for(std::vector<bool>::iterator it = m_slipDetectorState.begin(); it != m_slipDetectorState.end(); ++it) {
		m_liveSlipDetection = m_liveSlipDetection || *it;
	}
	// Acquire exclusive access and reset queue
	boost::unique_lock<boost::mutex> lock(m_mutexLiveFrame);
	m_slipResultQueue[matrixID] = boost::make_shared<std::queue<slipResult> > ();
	m_conditionLiveFrame.notify_all();
}


bool FrameManager::getSlipDetectionState() {
	return m_liveSlipDetection;
}


bool FrameManager::getSlipDetectionState(uint matrixID) {
	return m_slipDetectorState[matrixID];
}


void FrameManager::setSlipThresholdReference(double thresh) {
	m_threshSlipReference = thresh;
}


void FrameManager::setSlipThresholdConsecutive(double thresh) {
	m_threshSlipConsecutive = thresh;
}


bool FrameManager::setSlipReferenceFrameLive(uint matrixID) {
	matrixInfo &matrixInfo = getMatrixInfo(matrixID);
	//TSFrame* tsFrame = getCurrentFrame();
	TSFrame* tsFrame = getCurrentFilteredFrame();
	cv::Mat referenceFrame = cv::Mat(matrixInfo.cells_y, matrixInfo.cells_x, CV_32F, tsFrame->cells.data()+matrixInfo.texel_offset); // No copying
	return m_slipDetectors[matrixID]->setReferenceFrame(referenceFrame);
}


slipResult FrameManager::computeSlipLive(uint matrixID) {
	matrixInfo &matrixInfo = getMatrixInfo(matrixID);
	//TSFrame* tsFrame = getCurrentFrame();
	TSFrame* tsFrame = getCurrentFilteredFrame();
	cv::Mat currentFrame = cv::Mat(matrixInfo.cells_y, matrixInfo.cells_x, CV_32F, tsFrame->cells.data()+matrixInfo.texel_offset); // No copying
	return m_slipDetectors[matrixID]->computeSlip(currentFrame);
}


std::vector<boost::optional<slipResult> > FrameManager::getSlipLive() {
	// Acquire exclusive access
	boost::unique_lock<boost::mutex> lock(m_mutexLiveFrame);
	while(!m_liveSlipAvailable) { // Loop to catch spurious wake-ups
		m_conditionLiveFrame.wait(lock);
	}
	m_liveSlipAvailable = false;

	return m_liveSlip;
}


bool FrameManager::getSlipLiveBinary() {
	if(m_liveSlipDetection) {
		// Acquire exclusive access
		boost::unique_lock<boost::mutex> lock(m_mutexLiveFrame);
		while(!m_liveSlipAvailable) { // Loop to catch spurious wake-ups
			m_conditionLiveFrame.wait(lock);
		}
		m_liveSlipAvailable = false;

		for(uint m = 0; m < 6; m++) {
			if(m_liveSlip[m]) {
				slipResult &slip = *(m_liveSlip[m]);
				if(slip.successTranslation) {

					bool slipStateReference = fabs(slip.slipVectorReference_x) > m_threshSlipReference
							|| fabs(slip.slipVectorReference_y) > m_threshSlipReference;

					bool slipStateConsecutive = fabs(slip.slipVector_x) > m_threshSlipConsecutive
							|| fabs(slip.slipVector_y) > m_threshSlipConsecutive;

					if(slipStateReference || slipStateConsecutive) {
						return true;
					}

				}
			}
		}
	}
	return false;
}


void FrameManager::slipResultProducer() {

	// Acquire exclusive access
	boost::unique_lock<boost::mutex> lock(m_mutexLiveFrame);

	// Compute slip for specified matrices and push it on queue
	for(uint m = 0; m < m_slipDetectorState.size(); m++) {
		if(m_slipDetectorState[m]) {
			m_liveSlip[m] = boost::optional<slipResult> ( computeSlipLive(m) );
			m_slipResultQueue[m]->push( *(m_liveSlip[m]) );
			m_liveSlipAvailable = true;
		} else {
			m_liveSlip[m] = boost::optional<slipResult> (); // Empty
		}
	}
	//lock.unlock(); // RAII should do it anyway
	m_conditionLiveFrame.notify_all();
}


std::vector<boost::optional<slipResult> >  FrameManager::slipResultConsumer() {

	// Acquire exclusive access
	boost::unique_lock<boost::mutex> lock(m_mutexLiveFrame);

	// Blocked waiting if queue is completely empty
	bool allQueuesEmpty = true;
	for(uint m = 0; m < m_slipResultQueue.size(); m++) {
		allQueuesEmpty = allQueuesEmpty && m_slipResultQueue[m]->empty();
	}
	while(allQueuesEmpty) { // Loop to catch spurious wake-ups
		m_conditionLiveFrame.wait(lock);
		allQueuesEmpty = true;
		for(uint m = 0; m < m_slipResultQueue.size(); m++) {
			allQueuesEmpty = allQueuesEmpty && m_slipResultQueue[m]->empty();
		}
	}

	// Remove slip detection results from queue
	std::vector<boost::optional<slipResult> > slipResults;
	for(uint m = 0; m < m_slipResultQueue.size(); m++) {
		if(m_slipDetectorState[m]) {
			// Copy front element
			slipResults.push_back( boost::optional<slipResult> ( m_slipResultQueue[m]->front() ) );
			// Delete reference
			m_slipResultQueue[m]->pop();
		} else {
			slipResults.push_back( boost::optional<slipResult> () ); // Empty
		}
	}
	//lock.unlock(); // RAII should do it anyway
	return slipResults;
}


bool FrameManager::setSlipReferenceFrame(uint matrixID, uint frameID) {
	matrixInfo &matrixInfo = getMatrixInfo(matrixID);
	//TSFrame* tsFrame = getFrame(frameID);
	TSFrame* tsFrame = getFilteredFrame(frameID);
	cv::Mat referenceFrame = cv::Mat(matrixInfo.cells_y, matrixInfo.cells_x, CV_32F, tsFrame->cells.data()+matrixInfo.texel_offset); // No copying
	return m_slipDetectors[matrixID]->setReferenceFrame(referenceFrame);
}


slipResult FrameManager::computeSlip(uint matrixID, uint frameID) {
	matrixInfo &matrixInfo = getMatrixInfo(matrixID);
	//TSFrame* tsFrame = getFrame(frameID);
	TSFrame* tsFrame = getFilteredFrame(frameID);
	cv::Mat currentFrame = cv::Mat(matrixInfo.cells_y, matrixInfo.cells_x, CV_32F, tsFrame->cells.data()+matrixInfo.texel_offset); // No copying
	return m_slipDetectors[matrixID]->computeSlip(currentFrame);
}


const string& FrameManager::getProfileName() {
	return m_profileName;
}


void FrameManager::loadFrames(const string& filename) {
	resetOffline();
	loadProfile(filename);
	initFilter();
}


void FrameManager::storeFrames(const string& filename) {
	storeProfile(filename);
}


void FrameManager::print_tree(const ptree& pt, int level) {
	const string sep(2 * level, ' ');
	BOOST_FOREACH(const ptree::value_type &v, pt) {
		cout << sep << v.first << " : " << v.second.data() << "\n";
		print_tree(v.second, level + 1);
	}
}


void FrameManager::print_tree(const ptree& pt) {
	print_tree(pt, 0);
}


float FrameManager::hex2float(string& s) {
	uint32_t x;
	std::stringstream ss;
	ss << std::hex << s;
	ss >> x;
	float f = reinterpret_cast<float&>(x);
	return f;
}


string FrameManager::float2hex(float f) {
	uint32_t x = reinterpret_cast<uint32_t&>(f);
	std::stringstream ss;
	ss << std::hex << std::uppercase << std::setfill ('0') << std::setw(8) << x;
	return ss.str();
}


void FrameManager::decodeTSFrame(const string& hexdata, std::vector<float>& decoded_frame) {
	uint k = 0; // Index to cells
	for(uint i = 0; i < hexdata.length(); i += 8) {
		string hex_token = hexdata.substr(i, 8);
		float rle_unit = hex2float(hex_token);
		if(rle_unit < 0) { // consecutive zeros
			for(uint j = 0; j < fabs(rle_unit); j++) {
				decoded_frame[k] = 0.0;
				k++;
			}
		} else {
			decoded_frame[k] = rle_unit; // nonzero cell value
			k++;
		}
	}
	if (k != decoded_frame.size()) {
		cout << "RLE encoded frame contains " << k << " texels, but " << decoded_frame.size() << " are expected\n";
	}
}


void FrameManager::encodeTSFrame(const vector<float>& cells, string& hexdata) {
	float consecutiveZeros = 0.0;
	for(uint i = 0; i < cells.size(); i++) {
		if(cells[i] == 0.0) {
			consecutiveZeros += 1.0;
		} else {
			if(consecutiveZeros > 0) {
				hexdata += float2hex(-consecutiveZeros); // hex representation of negative number of consecutive zeros
				consecutiveZeros = 0;
			}
			hexdata += float2hex(cells[i]); // hex representation of cell value
		}
	}
	if(consecutiveZeros > 0) { // Don't forget trailing zeros
		hexdata += float2hex(-consecutiveZeros);
	}
}


void FrameManager::loadProfile(const string& filename) {
	std::ifstream infile(filename.c_str(), std::ios_base::in | std::ios_base::binary);
	if (!infile) {
		std::cerr<< "Error! Can't open file: " << filename << std::endl;
		return;
	}

	filtering_streambuf<input> bz2inStreamBuf;
	bz2inStreamBuf.push(bzip2_decompressor());
	bz2inStreamBuf.push(infile);
	std::istream inStream(&bz2inStreamBuf); // read_xml() expects a string stream

	// Read the XML file into the property tree
	ptree xmlTree;
	try {
		read_xml(inStream, xmlTree, xml_parser::trim_whitespace);
	}
	catch (std::exception &e) {
		std::string what = e.what();
		// Since we are reading from a stream instead of a file...
		boost::replace_first(what, "<unspecified file>", std::string("<") + filename + ">");
		std::cerr << "Error while parsing file: " << what << endl;
	}

	//print_tree(xmlTree);

	// Iterate over <sensor *> entries and add matrices
	uint nMatrices = 0;
	sensorInfo& sensor = getSensorInfo();

	BOOST_FOREACH(const ptree::value_type &v, xmlTree.get_child("root.sensor")) {
		if(v.first == "generated_by" ) {
			sensor.generated_by = v.second.get_value<uint>();
		} else if(v.first == "hw_revision" ) {
			sensor.hw_revision = v.second.get_value<uint>();
		} else if(v.first == "serial_nr" ) {
			string serial_hex = v.second.get_value<string>();
			// hex to int
			stringstream converter(serial_hex);
			converter >> std::hex >> sensor.serial_no;
		} else if(v.first == "num_matrices" ) {
			nMatrices = v.second.get_value<uint>();
		}else if(v.first == "converter_resolution" ) {
			sensor.converter_resolution = v.second.get_value<uint>();
		}
		else if(v.first == "matrix" ) {
			const ptree& frameNode = v.second;
			matrixInfo matrix;

			// TODO: parse one_wire_uid
			// Unique 48-bit value that identifies the matrix

			matrix.hw_revision = frameNode.get<uint>("hw_revision");
			matrix.cells_x = frameNode.get<uint>("cells_x");
			matrix.cells_y = frameNode.get<uint>("cells_y");
			matrix.num_cells = matrix.cells_x * matrix.cells_y;
			matrix.texel_width = frameNode.get<float>("texel_width");
			matrix.texel_height = frameNode.get<float>("texel_height");
			matrix.matrix_center_x = frameNode.get<float>("matrix_center_x");
			matrix.matrix_center_y = frameNode.get<float>("matrix_center_y");
			matrix.matrix_center_z = frameNode.get<float>("matrix_center_z");
			matrix.matrix_theta_x = frameNode.get<float>("theta_x");
			matrix.matrix_theta_y = frameNode.get<float>("theta_y");
			matrix.matrix_theta_z = frameNode.get<float>("theta_z");
			matrix.fullscale = frameNode.get<float>("fullscale");
			std::string staticMaskString = frameNode.get<string>("static_mask");
			std::string dynamicMaskString = frameNode.get<string>("dynamic_mask");

			// Remove all white spaces
			staticMaskString.erase(remove_if(staticMaskString.begin(), staticMaskString.end(), ::isspace), staticMaskString.end());
			dynamicMaskString.erase(remove_if(dynamicMaskString.begin(), dynamicMaskString.end(), ::isspace), dynamicMaskString.end());

			// Convert mask string to bool vector
			for(std::string::size_type i = 0; i < staticMaskString.size(); ++i) {
				staticMaskString[i] == '1' ? matrix.static_mask.push_back(true) : matrix.static_mask.push_back(false);
			}
			for(std::string::size_type i = 0; i < dynamicMaskString.size(); ++i) {
				dynamicMaskString[i] == '1' ? matrix.dynamic_mask.push_back(true) : matrix.dynamic_mask.push_back(false);
			}

			// Init slip detectors
			m_slipDetectors.push_back( boost::make_shared<SlipDetector>(matrix.cells_x, matrix.cells_y) );

			// Add sensor matrix
			matrix.texel_offset = m_sensor.nb_cells;
			m_matrices.push_back(matrix);
			m_sensor.nb_matrices++;
			m_sensor.nb_cells += matrix.num_cells;
		}
	}

	if(nMatrices != getNumMatrices()) {
		std::cerr << "Error: Invalid number of sensor matrices" << endl;
		std::cerr << nMatrices << "  " << getNumMatrices() << endl;
		exit(EXIT_FAILURE);
	}

	// Iterate over <data> entries
	uint nFrames = 0;
	BOOST_FOREACH(const ptree::value_type &v, xmlTree.get_child("root.data")) {
		if(v.first == "frame_count" ) {
			//cout << "Frame Count = " << v.second.data() << endl;
		} else {
			const ptree& frameNode = v.second;
			//int id = frameNode.get<int>("<xmlattr>.index");

			// allocate space for new tactile sensor frame
			TSFrame& tsframe = allocateTSFrame();
			tsframe.timestamp = frameNode.get<uint64_t>("time_stamp");

			string encoded_frame = frameNode.get<string>("frame_data");

			// Remove all white spaces from hexadecimal frame data (trim_whitespace works insufficiently)
			encoded_frame.erase(remove_if(encoded_frame.begin(), encoded_frame.end(), ::isspace), encoded_frame.end());

			// Decode frame (in-place)
			decodeTSFrame(encoded_frame, tsframe.cells);
			nFrames++;
		}
	}
	if(nFrames != getFrameCountTS()) {
		std::cerr << "Error: Invalid number of tactile sensor frames" << endl;
		std::cerr << nFrames << "  " << getFrameCountTS() << endl;
		exit(EXIT_FAILURE);
	}


	// Iterate over <temperatures> entries if present
	uint nTemps = 0;
	boost::optional<ptree &> temperatureNode = xmlTree.get_child_optional("root.temperatures");
	if(temperatureNode) {
		typedef boost::tokenizer<boost::escaped_list_separator<char> >  Token; // Default split: ","
		BOOST_FOREACH(const ptree::value_type &v, xmlTree.get_child("root.temperatures")) {
			if(v.first == "temp_count" ) {
				//cout << "Frame Count = " << v.second.data() << endl;
			} else {
				const ptree& tempNode = v.second;
				TemperatureFrame temperature;
				temperature.timestamp = tempNode.get<uint64_t>("time_stamp");
				string temperatureStr = tempNode.get<string>("temp_data");

				// Tokenize string
				Token tokens(temperatureStr);
				for(Token::iterator i = tokens.begin(); i != tokens.end(); ++i) {
					std::string token(*i);
					boost::trim(token);
					temperature.values.push_back(boost::lexical_cast<double>(token));
				}
				m_temperatureFrames.push_back(temperature);
				nTemps++;
			}
		}
		if(nTemps != getFrameCountTemperature()) {
			std::cerr << "Error: Invalid number of temperature frames" << endl;
			std::cerr << nTemps << "  " << getFrameCountTemperature() << endl;
			exit(EXIT_FAILURE);
		}
	}
	// Create mapping between tactile sensor frame and associated temperature
	createTemperatureMapping();

	// Iterate over <joint_angles> entries if present
	uint nAngles = 0;
	boost::optional<ptree &> jointAngleNode = xmlTree.get_child_optional("root.joint_angles");
	if(jointAngleNode) {
		typedef boost::tokenizer<boost::escaped_list_separator<char> >  Token; // Default split: ","
		BOOST_FOREACH(const ptree::value_type &v, xmlTree.get_child("root.joint_angles")) {
			if(v.first == "angle_count" ) {
				//cout << "Frame Count = " << v.second.data() << endl;
			} else {
				const ptree& angleNode = v.second;
				JointAngleFrame jointAngles;
				jointAngles.timestamp = angleNode.get<uint64_t>("time_stamp");
				string JointAngleStr = angleNode.get<string>("angle_data");

				// Tokenize string
				Token tokens(JointAngleStr);
				for(Token::iterator i = tokens.begin(); i != tokens.end(); ++i) {
					std::string token(*i);
					boost::trim(token);
					jointAngles.angles.push_back(boost::lexical_cast<double>(token));
				}
				m_jointAngleFrames.push_back(jointAngles);
				nAngles++;
			}
		}
		if(nAngles != getFrameCountJointAngles()) {
			std::cerr << "Error: Invalid number of joint angle frames" << endl;
			std::cerr << nAngles << "  " << getFrameCountJointAngles() << endl;
			exit(EXIT_FAILURE);
		}
	}
	if(nAngles > 0) {
		m_jointAngleFrameAvailable = true;
	} else {
		m_jointAngleFrameAvailable = false;
	}

	// Create mapping between tactile sensor frame and associated joint angles
	createJointAngleMapping();

	m_selectedCells.assign(getNumCells(), false); // Reset selected cells
	m_profileName = filename;
	m_tsFrameAvailable = true;

	cout << nFrames << " Tactile sensor frames, " << nTemps << " temperatures and " << nAngles << " joint angles loaded from " << filename << endl;
}


void FrameManager::storeProfile(const string& filename) {

	printf("Storing Profile!\n");

	std::ofstream outfile(filename.c_str(), std::ios_base::out | std::ios::binary);
	boost::iostreams::filtering_stream<output> bz2outStream;
	bz2outStream.push(boost::iostreams::bzip2_compressor());
	bz2outStream.push(outfile);
	ptree xmlTree;

	// Add sensor
	xmlTree.add("root.sensor.<xmlattr>.name", "");
	xmlTree.add("root.sensor.generated_by", 105); // Firmware revision
	xmlTree.add("root.sensor.hw_revision", 1);
	// int to hex
	stringstream converter;
	converter << std::setfill ('0') << std::setw(sizeof(UInt32)*2) << std::hex << std::uppercase << getSensorInfo().serial_no;
	string serial_hex(converter.str());
	xmlTree.add("root.sensor.serial_nr", serial_hex);
	xmlTree.add("root.sensor.num_matrices", getNumMatrices());
	xmlTree.add("root.sensor.converter_resolution", 12); // 12 Bit resolution of AD converter

	// Add matrices
	for(uint i = 0; i < getNumMatrices(); i++) {
		matrixInfo& matrix = getMatrixInfo(i);
		ptree& matrixNode = xmlTree.add("root.sensor.matrix", "");
		matrixNode.add("<xmlattr>.index", i);
		matrixNode.add("<xmlattr>.name", "");
		matrixNode.add("cells_x", matrix.cells_x);
		matrixNode.add("cells_y", matrix.cells_y);
		matrixNode.add("texel_width", matrix.texel_width);
		matrixNode.add("texel_height", matrix.texel_height);
		matrixNode.add("matrix_center_x", matrix.matrix_center_x);
		matrixNode.add("matrix_center_y", matrix.matrix_center_y);
		matrixNode.add("matrix_center_z", matrix.matrix_center_z);
		matrixNode.add("theta_x", matrix.matrix_theta_x);
		matrixNode.add("theta_y", matrix.matrix_theta_y);
		matrixNode.add("theta_z", matrix.matrix_theta_z);
		matrixNode.add("output_unit", "mV");
		matrixNode.add("hw_revision", matrix.hw_revision);
		matrixNode.add("one_wire_uid", "Not implemented yet"); // TODO: implement it!
		matrixNode.add("fullscale", matrix.fullscale);

		// Pretty printing of static mask
		std::string staticMaskString;
		// Convert mask string to bool vector
		for(uint i = 0; i < matrix.static_mask.size(); i++) {
			matrix.static_mask[i] == true ? staticMaskString += "1" : staticMaskString += "0";
		}
		uint pos = 0;
		std::string indentation(8, ' ');
		std::string linebreak = "\r\n";
		int linewidth = matrix.cells_x;
		std::string formated_static_mask;
		for(uint y = 0; y < matrix.cells_y; y++) {
			formated_static_mask += linebreak;
			formated_static_mask += indentation + staticMaskString.substr(pos, linewidth);
			pos += linewidth;
		}
		formated_static_mask += linebreak + std::string(6, ' ');
		matrixNode.add("static_mask", formated_static_mask);

		// Pretty printing of dynamic mask
		std::string dynamicMaskString;
		// Convert mask string to bool vector
		for(uint i = 0; i < matrix.dynamic_mask.size(); i++) {
			matrix.dynamic_mask[i] == true ? dynamicMaskString += "1" : dynamicMaskString += "0";
		}
		pos = 0;
		std::string formated_dynamic_mask;
		for(uint y = 0; y < matrix.cells_y; y++) {
			formated_dynamic_mask += linebreak;
			formated_dynamic_mask += indentation + dynamicMaskString.substr(pos, linewidth);
			pos += linewidth;
		}
		formated_dynamic_mask += linebreak + std::string(6, ' ');
		matrixNode.add("dynamic_mask", formated_dynamic_mask);
	}

	// Add frame data
	xmlTree.add("root.data.frame_count", getFrameCountTS());
	for(uint i = 0; i < getFrameCountTS(); i++) {
		ptree& frameNode = xmlTree.add("root.data.frame", "");
		frameNode.add("<xmlattr>.index", i);
		frameNode.add("time_stamp", getFrame(i)->timestamp);

		// Encode sensor frame
		string hexdata;
		encodeTSFrame(getFrame(i)->cells, hexdata);

		// Pretty printing of encoded frame
		uint pos = 0;
		uint linewidth = 8*7;
		string formated_hexdata;
		uint nLines = (uint)ceil((float)hexdata.length() / (float)linewidth);
		for(uint y = 0; y < nLines; y++) {
			formated_hexdata += "\r\n        ";
			formated_hexdata += hexdata.substr(pos, linewidth);
			pos += linewidth;
		}
		formated_hexdata += "\r\n      ";
		frameNode.add("frame_data", formated_hexdata);
	}

	// Add temperature readings
	xmlTree.add("root.temperatures.temp_count", getFrameCountTemperature());
	for(uint i = 0; i < getFrameCountTemperature(); i++) {
		ptree& frameNode = xmlTree.add("root.temperatures.temp", "");
		frameNode.add("<xmlattr>.index", i);
		frameNode.add("time_stamp", getTemperatureFrame(i)->timestamp);

		// Print temperatures
		std::stringstream temperatures;
		for(uint t = 0; t <  m_temperatureFrames[i].values.size()-1; t++ ) {
			temperatures << setprecision(2) << std::fixed << m_temperatureFrames[i].values[t] << ", ";
		}
		temperatures << setprecision(2) << std::fixed << m_temperatureFrames[i].values.back();

		frameNode.add("temp_data", temperatures.str());
	}

	// Add joint angle readings
	xmlTree.add("root.joint_angles.angle_count", getFrameCountJointAngles());
	for(uint i = 0; i < getFrameCountJointAngles(); i++) {
		ptree& frameNode = xmlTree.add("root.joint_angles.angle", "");
		frameNode.add("<xmlattr>.index", i);
		frameNode.add("time_stamp", getJointAngleFrame(i)->timestamp);

		// Print joint angles
		std::stringstream jointAngles;
		for(uint t = 0; t <  m_jointAngleFrames[i].angles.size()-1; t++ ) {
			jointAngles << setprecision(5) << std::fixed << m_jointAngleFrames[i].angles[t] << ", ";
		}
		jointAngles << setprecision(5) << std::fixed << m_jointAngleFrames[i].angles.back();

		frameNode.add("angle_data", jointAngles.str());
	}

	write_xml(bz2outStream, xmlTree, xml_writer_settings<char>(' ', 2));

	cout << getFrameCountTS() << " Frames saved to " << filename << endl;
}


void FrameManager::storeProfileSelection(const string& filename, uint fromIdxTS, uint toIdxTS) {

	uint64_t timestampFrom = getFrame(fromIdxTS)->timestamp;
	uint64_t timestampTo = getFrame(toIdxTS)->timestamp;

	// Determine corresponding temperature frames
	std::deque<TemperatureFrame>::iterator itFromTemp;
	std::deque<TemperatureFrame>::iterator itToTemp;
	itFromTemp = std::lower_bound(m_temperatureFrames.begin(), m_temperatureFrames.end(), timestampFrom, TimestampComparator<TemperatureFrame>() ); // Position right before timestampFrom
	itToTemp = std::upper_bound(m_temperatureFrames.begin(), m_temperatureFrames.end(), timestampTo, TimestampComparator<TemperatureFrame>() ); // Position right after timestampTo
	uint fromIdxTemp = itFromTemp - m_temperatureFrames.begin();
	uint toIdxTemp = itToTemp - m_temperatureFrames.begin(); // off by one

	// Determine corresponding joint angle frames
	std::deque<JointAngleFrame>::iterator itFromJointAngle;
	std::deque<JointAngleFrame>::iterator itToJointAngle;
	itFromJointAngle = std::lower_bound(m_jointAngleFrames.begin(), m_jointAngleFrames.end(), timestampFrom, TimestampComparator<JointAngleFrame>() ); // Position right before timestampFrom
	itToJointAngle = std::upper_bound(m_jointAngleFrames.begin(), m_jointAngleFrames.end(), timestampTo, TimestampComparator<JointAngleFrame>() ); // Position right after timestampTo
	uint fromIdxJointAngle = itFromJointAngle - m_jointAngleFrames.begin();
	uint toIdxJointAngle = itToJointAngle - m_jointAngleFrames.begin(); // off by one

	// Including both limits
	uint numFramesTS = toIdxTS - fromIdxTS + 1;
	uint numFramesTemp = toIdxTemp - fromIdxTemp;
	uint numFramesJointAngle = toIdxJointAngle - fromIdxJointAngle;

	std::ofstream outfile(filename.c_str(), std::ios_base::out | std::ios::binary);
	boost::iostreams::filtering_stream<output> bz2outStream;
	bz2outStream.push(boost::iostreams::bzip2_compressor());
	bz2outStream.push(outfile);
	ptree xmlTree;

	// Add sensor
	xmlTree.add("root.sensor.<xmlattr>.name", "");
	xmlTree.add("root.sensor.generated_by", 105); // Firmware revision
	xmlTree.add("root.sensor.hw_revision", 1);
	// int to hex
	stringstream converter;
	converter << std::setfill ('0') << std::setw(sizeof(UInt32)*2) << std::hex << std::uppercase << getSensorInfo().serial_no;
	string serial_hex(converter.str());
	xmlTree.add("root.sensor.serial_nr", serial_hex);
	xmlTree.add("root.sensor.num_matrices", getNumMatrices());
	xmlTree.add("root.sensor.converter_resolution", 12); // 12 Bit resolution of AD converter

	// Add matrices
	for(uint i = 0; i < getNumMatrices(); i++) {
		matrixInfo& matrix = getMatrixInfo(i);
		ptree& matrixNode = xmlTree.add("root.sensor.matrix", "");
		matrixNode.add("<xmlattr>.index", i);
		matrixNode.add("<xmlattr>.name", "");
		matrixNode.add("cells_x", matrix.cells_x);
		matrixNode.add("cells_y", matrix.cells_y);
		matrixNode.add("texel_width", matrix.texel_width);
		matrixNode.add("texel_height", matrix.texel_height);
		matrixNode.add("matrix_center_x", matrix.matrix_center_x);
		matrixNode.add("matrix_center_y", matrix.matrix_center_y);
		matrixNode.add("matrix_center_z", matrix.matrix_center_z);
		matrixNode.add("theta_x", matrix.matrix_theta_x);
		matrixNode.add("theta_y", matrix.matrix_theta_y);
		matrixNode.add("theta_z", matrix.matrix_theta_z);
		matrixNode.add("output_unit", "mV");
		matrixNode.add("hw_revision", matrix.hw_revision);
		matrixNode.add("one_wire_uid", "Not implemented yet"); // TODO: implement it!
		matrixNode.add("fullscale", matrix.fullscale);

		// Pretty printing of static mask
		std::string staticMaskString;
		// Convert mask string to bool vector
		for(uint i = 0; i < matrix.static_mask.size(); i++) {
			matrix.static_mask[i] == true ? staticMaskString += "1" : staticMaskString += "0";
		}
		uint pos = 0;
		std::string indentation(8, ' ');
		std::string linebreak = "\r\n";
		int linewidth = matrix.cells_x;
		std::string formated_static_mask;
		for(uint y = 0; y < matrix.cells_y; y++) {
			formated_static_mask += linebreak;
			formated_static_mask += indentation + staticMaskString.substr(pos, linewidth);
			pos += linewidth;
		}
		formated_static_mask += linebreak + std::string(6, ' ');
		matrixNode.add("static_mask", formated_static_mask);

		// Pretty printing of dynamic mask
		std::string dynamicMaskString;
		// Convert mask string to bool vector
		for(uint i = 0; i < matrix.dynamic_mask.size(); i++) {
			matrix.dynamic_mask[i] == true ? dynamicMaskString += "1" : dynamicMaskString += "0";
		}
		pos = 0;
		std::string formated_dynamic_mask;
		for(uint y = 0; y < matrix.cells_y; y++) {
			formated_dynamic_mask += linebreak;
			formated_dynamic_mask += indentation + dynamicMaskString.substr(pos, linewidth);
			pos += linewidth;
		}
		formated_dynamic_mask += linebreak + std::string(6, ' ');
		matrixNode.add("dynamic_mask", formated_dynamic_mask);
	}

	// Add tactile sensor frame data
	xmlTree.add("root.data.frame_count", numFramesTS);
	for(uint i = fromIdxTS; i <= toIdxTS; i++) {
		ptree& frameNode = xmlTree.add("root.data.frame", "");
		frameNode.add("<xmlattr>.index", i-fromIdxTS);
		frameNode.add("time_stamp", getFrame(i)->timestamp);

		// Encode sensor frame
		string hexdata;
		encodeTSFrame(getFrame(i)->cells, hexdata);

		// Pretty printing of encoded frame
		uint pos = 0;
		uint linewidth = 8*7;
		string formated_hexdata;
		uint nLines = (uint)ceil((float)hexdata.length() / (float)linewidth);
		for(uint y = 0; y < nLines; y++) {
			formated_hexdata += "\r\n        ";
			formated_hexdata += hexdata.substr(pos, linewidth);
			pos += linewidth;
		}
		formated_hexdata += "\r\n      ";
		frameNode.add("frame_data", formated_hexdata);
	}

	// Add temperature readings
	xmlTree.add("root.temperatures.temp_count", numFramesTemp);
	for(uint i = fromIdxTemp; i < toIdxTemp; i++) {
		ptree& frameNode = xmlTree.add("root.temperatures.temp", "");
		frameNode.add("<xmlattr>.index", i-fromIdxTemp);
		frameNode.add("time_stamp", getTemperatureFrame(i)->timestamp);

		// Print temperatures
		std::stringstream temperatures;
		for(uint t = 0; t <  m_temperatureFrames[i].values.size()-1; t++ ) {
			temperatures << setprecision(2) << std::fixed << m_temperatureFrames[i].values[t] << ", ";
		}
		temperatures << setprecision(2) << std::fixed << m_temperatureFrames[i].values.back();

		frameNode.add("temp_data", temperatures.str());
	}

	// Add joint angle readings
	xmlTree.add("root.joint_angles.angle_count", numFramesJointAngle);
	for(uint i = fromIdxJointAngle; i < toIdxJointAngle; i++) {
		ptree& frameNode = xmlTree.add("root.joint_angles.angle", "");
		frameNode.add("<xmlattr>.index", i-fromIdxJointAngle);
		frameNode.add("time_stamp", getJointAngleFrame(i)->timestamp);

		// Print joint angles
		std::stringstream jointAngles;
		for(uint t = 0; t <  m_jointAngleFrames[i].angles.size()-1; t++ ) {
			jointAngles << setprecision(5) << std::fixed << m_jointAngleFrames[i].angles[t] << ", ";
		}
		jointAngles << setprecision(5) << std::fixed << m_jointAngleFrames[i].angles.back();

		frameNode.add("angle_data", jointAngles.str());
	}

	write_xml(bz2outStream, xmlTree, xml_writer_settings<char>(' ', 2));

	cout << numFramesTS << " Frames saved to " << filename << endl;
}


void FrameManager::printJointAngleFrame(JointAngleFrame& jointAngleFrame) {
	cout << "Joint Angle: ";
	for(uint i = 0; i < jointAngleFrame.angles.size(); i++ ) {
		cout << setw(6) << setprecision(3) << fixed << jointAngleFrame.angles[i] << " ";
	}
	cout << endl;
}


void FrameManager::printTemperatureFrame(TemperatureFrame& temperatureFrame) {
	cout << "Temperatures: ";
	for(uint i = 0; i < temperatureFrame.values.size(); i++ ) {
		cout << setw(6) << setprecision(2) << fixed << temperatureFrame.values[i];
	}
	cout << endl;
}


void FrameManager::printTSMatrix(uint frameID, uint m) {
	cout <<  "Matrix " << m << ":\n";
	for(uint y = 0; y < m_matrices[m].cells_y; y++) {
		cout << std::setw( 2 ) << y << "| ";
		for(uint x = 0; x < m_matrices[m].cells_x; x++) {
			cout << std::setw( 4 ) << getTexel(frameID, m, x, y) << " ";
		}
		cout << endl;
	}
}


void FrameManager::printTSMatrices(uint frameID) {
	for(uint m = 0; m < m_sensor.nb_matrices; m++) {
		printTSMatrix(frameID, m);
		cout << endl;
	}
}
