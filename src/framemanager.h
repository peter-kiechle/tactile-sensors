#ifndef FRAMEMANAGER_HPP_
#define FRAMEMANAGER_HPP_

#include <deque>
#include <vector>
#include <queue>

#include <math.h>
#include <stdint.h>

#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/optional.hpp>

#include "sdh/sdh.h"
#include "sdh/dsa.h"

#include "frameprocessor.h"
#include "slipdetection.h"

using namespace std;
USING_NAMESPACE_SDH


/// Tactile sensor Controller info (see dsa.h)
struct sensorInfo {
	uint nb_matrices; // number of sensor matrices
	uint nb_cells; // total number of sensor cells
	uint generated_by; // Firmware revision used to set up the sensor configuration
	uint hw_revision; // Revision number of the measurement transducer
	uint serial_no; // Serial number of the distributed measurement transducer
	uint converter_resolution;
};


/// Individual sensor matrices (see dsa.h)
struct matrixInfo {
	UInt8 uid[6]; // Unique 48-bit value that identifies the matrix
	uint hw_revision; // Revision number (xx.xx) of the measurement transducer
	uint cells_x; // Number of horizontal texels
	uint cells_y; // Number of vertical texels
	float texel_width; // Width of one texel in millimeters
	float texel_height; // Width of one texel in millimeters
	float matrix_center_x; // X-coordinate of the matrix center (in mm)
	float matrix_center_y; // Y-coordinate of the matrix center (in mm)
	float matrix_center_z; // Z-coordinate of the matrix center (in mm)
	float matrix_theta_x; // Rotation of matrix about x-axis (in degrees)
	float matrix_theta_y; // Rotation of matrix about y-axis (in degrees)
	float matrix_theta_z; // Rotation of matrix about z-axis (in degrees)
	float fullscale; // Fullscale value of the output signal. For 12-Bit converters: 4095
	std::vector<bool> static_mask; // Factory set and not editable
	std::vector<bool> dynamic_mask; // Selected texel of interest for faster sampling
	uint num_cells; // cells_x * cells_y
	uint texel_offset; // Offset to the first texel of each matrix in the entire frame
};


/// Tactile sensor frame
struct TSFrame {
	std::vector<float> cells;
	uint64_t timestamp;
	TSFrame() {}
	TSFrame(uint nb_cells) : cells(nb_cells) {}
};


/// Temperatures 0-6: close to axes motors, Temperature 7: FPGA, Temperature 8: Printed circuit board
struct TemperatureFrame {
	std::vector<double> values;
	uint64_t timestamp;
};


/// Joint angles in accordance with SDHLibrary
struct JointAngleFrame {
	std::vector<double> angles;
	uint64_t timestamp;
};


/// Timestamp comparator functor
template <typename Frame>
struct TimestampComparator {
	bool operator()(const Frame& frame1, const Frame& frame2) const {
		return frame1.timestamp < frame2.timestamp;
	}
	bool operator()(const Frame& frame, uint64_t timestamp) const {
		return frame.timestamp < timestamp;
	}
	bool operator()(uint64_t timestamp, const Frame& frame) const {
		return timestamp < frame.timestamp;
	}
};


class FrameIO; // Forward declaration
class FrameGrabberDSA; // Forward declaration
class FrameGrabberSDH; // Forward declaration


/**
 * @class FrameManager
 * @brief The heart of this project.
 *
 * @note
 * 	A word on copy constructors:
 *	Since boost::mutex and boost::condition_variable are not copyable simply delete the copy constructor.
 *	Alternative: write a copy constructor that copies the data but constructs a new mutex,
 *	i.e. FrameManager(const FrameManager&) = delete;
 *
 */
class FrameManager {

private:
	cSDH *hand;
	cDSA *ts;
	FrameGrabberDSA *frameGrabberDSA;
	FrameGrabberSDH *frameGrabberSDH;
	FrameProcessor frameProcessor;

	uint m_currentFrameID; // Should have been size_t, but it's too late for that now... (On a 32 bit machine, you can continuously record for 5 years)
	uint m_currentFilteredFrameID; // see m_currentFrameID

	sensorInfo m_sensor; // Holds info about the sensor controller
	std::vector<matrixInfo> m_matrices; // Holds info about the sensor matrices

	boost::shared_ptr<TSFrame> m_liveFrame; // For real-time applications. Latest frame received (yet another copy of the receive buffer, but with float values)
	boost::shared_ptr<TSFrame> m_filteredFrame; // Yet another copy for the purpose of filtering

	JointAngleFrame liveJointAngleFrame; // For real-time applications

	std::deque<TSFrame> m_tsframes; // Double ended queue of tactile sensor frames
	std::deque<TemperatureFrame> m_temperatureFrames; // Double ended queue of temperature readings
	std::deque<JointAngleFrame> m_jointAngleFrames; // Double ended queue of joint angle readings

	std::vector<uint> m_temperatureMapping; // Mapping between tactile sensor frame and associated temperature: mapping[frameID] = tempID
	std::vector<uint> m_jointAngleMapping; // Mapping between tactile sensor frame and associated joint angles: mapping[frameID] = angleID

	std::vector<float> m_matrixSensitivity; // Sensitivity settings of matrices 0-5
	std::vector<UInt16> m_matrixThreshold; // Threshold of matrices 0-5

	std::string m_profileName; // The *dsa file's name

	bool m_tsFrameAvailable; // Tactile sensor frame is available
	bool m_jointAngleFrameAvailable; // Joint angle frame is available

	std::vector<bool> m_selectedCells; // Mask of selected cells

	// Slip detection
	std::vector<boost::shared_ptr<SlipDetector> > m_slipDetectors; // Per matrix slip-detection
	std::vector<bool> m_slipDetectorState; // Per matrix slip-state
	boost::mutex m_mutexLiveFrame; // For thread safe access to live frames
	boost::condition_variable m_conditionLiveFrame; // For thread safe access to live frames
	bool m_liveSlipDetection; // Live mode
	bool m_liveSlipAvailable; // Live mode
	double m_threshSlipConsecutive; // Threshold for comparison with previous matrix
	double m_threshSlipReference; // Threshold for comparison with reference matrix
	std::vector<boost::optional<slipResult> > m_liveSlip; // Results of last slip computation
	std::vector<boost::shared_ptr<std::queue<slipResult> > > m_slipResultQueue; // Queue of slip results (producer consumer pattern).

	void initFilter();

public:

	/**
	 * Constructor.
	 * Calls resetOffline().
	 */
	FrameManager();

	virtual ~FrameManager();


	/**
	 * Resets frame manager (offline state).
	 * @return void
	 */
	void resetOffline();


	/**
	 * Resets frame manager (online state).
	 * @return void
	 */
	void resetOnline();


	/**
	 * Sets the SDH-2
	 * @param sdh The hand instance.
	 * @return void
	 */
	void setSDH(cSDH *sdh);


	/**
	 * Sets the DSA-controller.
	 * Calls queryDSAInfo() and initializes frame manager accordingly.
	 * @param dsa The DSA instance.
	 * @return void
	 */
	void setDSA(cDSA *dsa);


	/**
	 * Checks if the hand is connected.
	 * @return The state.
	 */
	bool isConnectedSDH();


	/**
	 * Checks if the DSA-controller is connected.
	 * @return The state.
	 */
	bool isConnectedDSA();


	/**
	 * Queries sensor controller and matrix info.
	 * @return void
	 */
	void queryDSAInfo();


	/**
	 * Sets the sensitivity of the specified matrix.
	 * @param matrixID The matrix ID.
	 * @param sensitivity The matrix sensitivity threshold in range [0.0, 1.0].
	 * @return void
	 */
	void setSensitivity(uint matrixID, float sensitivity);

	/**
	 * Sets the sensor value threshold.
	 * @note Program might crash if the connection is choking due to the failing run length encoding in case of noise.
	 *       This problem is hidden somewhere in the SDH-2's black-box.
	 * @param matrixID The matrix ID.
	 * @param threshold The matrix threshold.
	 * @return void
	 */
	void setThreshold(uint matrixID, float threshold);


	/**
	 * Sets the DSA frame grabber.
	 * @param fgDSA The DSA frame grabber.
	 * @return void
	 */
	void setFrameGrabberDSA(FrameGrabberDSA *fgDSA);


	/**
	 * Sets the SDH-2 frame grabber.
	 * @param fgSDH The SDH frame grabber.
	 * @return void
	 */
	void setFrameGrabberSDH(FrameGrabberSDH *fgSDH);


	/**
	 * Gets the queried sensor info.
	 * @return Reference to sensorInfo.
	 */
	inline sensorInfo& getSensorInfo() { return m_sensor; }


	/**
	 * Gets the queried matrix info.
	 * @param i The matrix.
	 * @return Reference to matrixInfo.
	 */
	inline matrixInfo& getMatrixInfo(uint i) { return m_matrices[i]; }


	/**
	 * Returns the number of matrices.
	 * @return The number of matrices.
	 */
	inline uint getNumMatrices() { return m_sensor.nb_matrices; }


	/**
	 * Returns the number of taxels.
	 * @return The number of taxels.
	 */
	inline uint getNumCells() { return m_sensor.nb_cells; }


	/**
	 * Returns the current frame ID.
	 * @return The current frame ID.
	 */
	uint getCurrentFrameID();


	/**
	 * Sets the current frame ID.
	 * @param frameID The current frame ID.
	 * @return void
	 */
	void setCurrentFrameID(uint frameID);


	/**
	 * Checks if a tactile sensor frame is available.
	 * @return The state.
	 */
	bool getTSFrameAvailable();


	/**
	 * Sets the availability of a tactile sensor frame.
	 * @param The state.
	 * @return void
	 */
	void setTSFrameAvailable(bool value);


	/**
	 * Checks if a joint angle frame is available.
	 * @return The state.
	 */
	bool getJointAngleFrameAvailable();


	/**
	 * Sets the availability of the joint angle frames.
	 * @param The state.
	 * @return void
	 */
	void setJointAngleFrameAvailable(bool value);


	/**
	 * Creates a copy of the current frame, the live frame.
	 * Access to live frame has to be synchronized.
	 * @return void
	 */
	void setLiveFrame();


	/**
	 * Allocates space for new tactile sensor frame on the queue and returns reference.
	 * @return Reference to the allocated tactile sensor frame.
	 */
	TSFrame& allocateTSFrame();


	/**
	 * Adds the current frame to the record
	 * @return void
	 */
	void addTSFrame();


	/**
	 * Deletes a single TSFrame without deleting corresponding temperatures/angles
	 * Use deleteTSFrames(from, to) to remove multiple frames.
	 * @note Operation might be expensive due to the used queue data structure.
	 * @param frameID The frame ID.
	 * @return void
	 */
	void deleteTSFrame(uint frameID);


	/**
	 * Trim frames (tactile sensor, temperature, joint angles) to selection [from to] including borders.
	 * @param timestamp_from From.
	 * @param timestamp_to To.
	 * @return void
	 */
	void cropToFrames(uint64_t timestamp_from, uint64_t timestamp_to);


	/**
	 * Get specified frame of recorded frame history.
	 * @param frameID The frame ID.
	 * @return Pointer to the tactile sensor frame.
	 */
	TSFrame* getFrame(uint frameID);


	/**
	 * Get current frame of recorded frame history.
	 * @return Pointer to the tactile sensor frame.
	 */
	TSFrame* getCurrentFrame();


	/**
	 * Get specified frame of recorded frame history (specified filter is applied).
	 * @param frameID The frame ID.
	 * @return Pointer to the tactile sensor frame.
	 */
	TSFrame* getFilteredFrame(uint frameID);

	/**
	 * Get current frame of recorded frame history (specified filter is applied).
	 * @return Pointer to the tactile sensor frame.
	 */
	TSFrame* getCurrentFilteredFrame();


	/**
	 * Returns taxel value of specified frame and cell ID.
	 * @param frameID The frame ID.
	 * @param cellID The cell ID in range [0, 486]
	 * @return The taxel value.
	 */
	float getTexel(uint frameID, uint cellID);


	/**
	 * Returns taxel value of specified frame, matrix and coordinate.
	 * @param frameID
	 * @param m The matrix ID.
	 * @param x,y Taxel coordinates.
	 * @return The taxel value.
	 */
	float getTexel(uint frameID, uint m, uint x, uint y);


	/**
	 * Returns filtered taxel value of specified frame and cell ID.
	 * @param frameID The frame ID.
	 * @param cellID The cell ID in Range [0, 486]
	 * @return The taxel value.
	 */
	float getFilteredTexel(uint frameID, uint cellID);


	/**
	 * Returns filtered taxel value of specified frame, matrix and coordinate.
	 * @param frameID
	 * @param m The matrix ID.
	 * @param x,y Taxel coordinates.
	 * @return The taxel value.
	 */
	float getFilteredTexel(uint frameID, uint m, uint x, uint y);


	/**
	 * Returns the number of recorded tactile sensor frames.
	 * @return The tactile sensor frame count.
	 */
	uint getFrameCountTS();


	/**
	 * Returns the number of recorded temperature readings.
	 * @return The temperature frame count.
	 */
	uint getFrameCountTemperature();


	/**
	 * Returns the number of recorded joint angle readings.
	 * @return The joint angle frame count.
	 */
	uint getFrameCountJointAngles();


	/**
	 * Requests a temperature frame from the connected SDH-2.
	 * @details Temperatures 0-6: close to axes motors, Temperature 7: FPGA, Temperature 8: Printed circuit board.
	 * @param record Should the frame be stored?
	 */
	void requestTemperatureFrame(bool record);


	/**
	 * Returns specified temperature frame.
	 * @param tempID The temperature frame ID.
	 * @return Pointer to the temperature frame.
	 */
	TemperatureFrame* getTemperatureFrame(uint tempID);


	/**
	 * Requests a joint angle frame from the connected SDH-2.
	 * @details 0 : common base axis of finger 0 and 2
	 *          1 : proximal axis of finger 0
	 *          2 : distal axis of finger 0
	 *          3 : proximal axis of finger 1
	 *          4 : distal axis of finger 1
	 *          5 : proximal axis of finger 2
	 *          6 : distal axis of finger 2
	 * @param record Should the frame be stored?
	 */
	void requestJointAngleFrame(bool record);


	/**
	 * Returns specified joint angle frame.
	 * @param angleID The joint angle frame ID.
	 * @return Pointer to the joint angle frame.
	 */
	JointAngleFrame* getJointAngleFrame(uint angleID);


	/**
	 * Returns current joint angle frame.
	 * @return Pointer to the joint angle frame.
	 */
	JointAngleFrame* getCurrentJointAngleFrame();


	/**
	 * Creates a mapping between tactile sensor frame and associated temperature.
	 * @return void
	 */
	void createTemperatureMapping();


	/**
	 * Creates a mapping between tactile sensor frame and associated joint angles.
	 * @return void
	 */
	void createJointAngleMapping();


	/**
	 * Given a tactile sensor frame, returns the associated temperature frame.
	 * @param frameID The frame ID.
	 * @return Pointer to the recorded temperature frame.
	 */
	TemperatureFrame* getCorrespondingTemperature(uint frameID);


	/**
	 * Given a tactile sensor frame, returns the associated joint angle frame.
	 * @param frameID The frame ID.
	 * @return Pointer to the recorded joint angle frame.
	 */
	JointAngleFrame* getCorrespondingJointAngle(uint frameID);


	/**
	 * Convert cellID of entire frame to (matrixID, x, y)
	 * @param cellID The cell ID in range [0, 486]
	 * @param m The matrix ID.
	 * @param x,y The Taxel coordinates.
	 * @return void
	 */
	inline void convertCellIndex(uint cellID, uint& m, uint& x, uint& y) {
		// Find corresponding matrix
		for(m = 0; m < getNumMatrices(); m++) {
			if(cellID < m_matrices[m].texel_offset ) {
				break;
			}
		}
		m--;
		uint cellID_matrix = cellID - m_matrices[m].texel_offset;
		x = cellID_matrix % m_matrices[m].cells_x;
		y = cellID_matrix / m_matrices[m].cells_x;
	}


	/**
	 * Convert (matrixID, x, y) to cellID of entire frame
	 * @param m The matrix ID.
	 * @param x,y The Taxel coordinates.
	 * @return The cellID
	 */
	inline uint convertCellIndex(uint m, uint x, uint y) {
		return m_matrices[m].texel_offset + y * m_matrices[m].cells_x + x;
	}


	/**
	 * Select specified taxel.
	 * @details see convertCellIndex().
	 * @param cellID
	 * @param value
	 * @return void.
	 */
	void selectCell(uint cellID, bool value);


	/**
	 * Checks if the specified taxel is selected.
	 * @param cellID The cell ID.
	 * @return The selection state.
	 */
	bool isSelected(int cellID);


	/**
	 * Returns the number of selected taxels.
	 * @return The number of selected taxels.
	 */
	int getNumSelectedCells();


	/**
	 * Returns a mask of selected taxels.
	 * @return Reference to mask.
	 */
	std::vector<bool>& getSelection();


	/**
	 * Creates a list of indices of selected taxels.
	 * @return The vector containing cell IDs of selected taxels.
	 */
	std::vector<int> createSelectedCellsIdx();


	/**
	 * Sets dynamic bitmask of selected taxels.
	 * @param bitmask Bitmask of active taxels.
	 */
	void setDynamicMask(std::vector<bool>& bitmask);


	/**
	 * Gets static mask of specified taxel.
	 * @param m The matrix ID.
	 * @param x,y The taxel coordinates.
	 * @return The masking state.
	 */
	bool getStaticMask(uint m, uint x, uint y);


	/**
	 * Gets dynamic mask of specified taxel.
	 * @param m The matrix ID.
	 * @param x,y The taxel coordinates.
	 * @return The masking state.
	 */
	bool getDynamicMask(uint m, uint x, uint y);


	/**
	 * Gets the frame processor.
	 * @return The frame processor.
	 */
	FrameProcessor* getFrameProcessor();


	/**
	 * Disable filtering.
	 * @return void
	 */
	void setFilterNone();


	/**
	 * Enables the 2D Median filter. Just a wrapper around frame processor.
	 * @param kernelRadius The filtering kernel' radius.
	 * @param masked Taxels that survived the filtering process retain their original values.
	 * @return void
	 */
	void setFilterMedian(int kernelRadius, bool masked);


	/**
	 * Enables the spatio-temporal 3x3x3 Median filter. Just a wrapper around frame processor.
	 * @param masked Taxels that survived the filtering process retain their original values.
	 * @return void
	 */
	void setFilterMedian3D(bool masked);


	/**
	 * Enables the Gaussian filter. Just a wrapper around frame processor.
	 * @param kernelRadius The filtering kernel' radius.
	 * @param sigma The standard deviation.
	 * @param borderType OpenCV border type.
	 * @return void
	 */
	void setFilterGaussian(int kernelRadius, double sigma, int borderType);


	/**
	 * Enables the Bilateral filter. Just a wrapper around frame processor.
	 * @param kernelRadius The filtering kernel' radius.
	 * @param sigmaColor The color standard deviation parameter.
	 * @param sigmaSpace The spatial standard deviation parameter.
	 * @param borderType OpenCV border type.
	 * @return void
	 */
	void setFilterBilateral(int kernelSize, double sigmaColor, double sigmaSpace, int borderType);


	/**
	 * Enables the opening operation. Just a wrapper around frame processor.
	 * @param kernelType OpenCV kernel type.
	 * @param kernelRadius The filtering kernel' radius.
	 * @param masked Taxels that survived the filtering process retain their original values.
	 * @param borderType OpenCV border type.
	 * @return void
	 */
	void setFilterMorphological(int kernelType, int kernelRadius, bool masked, int borderType);


	/**
	 * Returns the slip-detector of specified matrix.
	 * @param matrixID The matrix ID.
	 * @return The slip-detector.
	 */
	boost::shared_ptr<SlipDetector> getSlipDetector(uint matrixID);


	/**
	 * Enables slip-detection on specified sensor matrix.
	 * @param matrixID The matrix ID.
	 */
	void enableSlipDetection(uint matrixID);


	/**
	 * Disables slip-detection on specified sensor matrix.
	 * @param matrixID The matrix ID.
	 */
	void disableSlipDetection(uint matrixID);


	/**
	 * Checks the combined slip-state.
	 * @return The slip-state.
	 */
	bool getSlipDetectionState();


	/**
	 * Checks the slip-state on specified sensor matrix.
	 * @param matrixID The matrix ID.
	 * @return The slip-state.
	 */
	bool getSlipDetectionState(uint matrixID);


	/**
	 * Sets the threshold for comparison with reference sensor matrix.
	 * @param thresh The theshold.
	 * @return void
	 */
	void setSlipThresholdReference(double thresh);


	/**
	 * Sets the threshold for comparison with previous sensor matrix.
	 * @param thresh The theshold.
	 * @return void
	 */
	void setSlipThresholdConsecutive(double thresh);


	/**
	 * (Re)sets the reference frame for both translational and rotational slip-detection (live version).
	 * @param matrixID The matrix ID.
	 * @return Success.
	 */
	bool setSlipReferenceFrameLive(uint matrixID);


	/**
	 * Performs both translational and rotational slip-detection (live version).
	 * @param matrixID The matrix ID.
	 * @return The combined results of the rotational and translational slip-detection.
	 */
	slipResult computeSlipLive(uint matrixID);


	/**
	 * Returns the results of the last slip computation (live version).
	 * @note Note: Might lead to a deadlock in combination with getSlipLiveBinary()
	 * @return The combined results.
	 */
	std::vector<boost::optional<slipResult> > getSlipLive();


	/**
	 * Computes a binary slip indicator from the slip results and the given thresholds
	 * @note Note: Might lead to a deadlock in combination with getSlipLive()
	 * @return The combined slip state.
	 */
	bool getSlipLiveBinary();


	/**
	 * Computes slip detection results and pushes them on queue. (producer-consumer pattern)
	 * @return void
	 */
	void slipResultProducer();


	/**
	 * Removes slip detection results from queue. (producer-consumer pattern)
	 * @return The last slip-detection result.
	 */
	std::vector<boost::optional<slipResult> > slipResultConsumer();


	/**
	 * (Re)sets the reference frame for both translational and rotational slip-detection (offline version).
	 * @param matrixID THe matrix ID.
	 * @param frameID The frame ID.
	 * @return Success.
	 */
	bool setSlipReferenceFrame(uint matrixID, uint frameID);


	/**
	 * Performs both translational and rotational slip-detection (offline version).
	 * @param matrixID The matrix ID.
	 * @param frameID The frame ID.
	 * @return The combined results of the rotational and translational slip-detection.
	 */
	slipResult computeSlip(uint matrixID, uint frameID);


	/**
	 * Gets the pressure profile's file name.
	 * @return The file name.
	 */
	const std::string& getProfileName();


	/**
	 * Load pressure profile from file.
	 * @param filename The *.dsa file.
	 * @return void
	 */
	void loadFrames(const std::string& filename);


	/**
	 * Store pressure profile.
	 * @param filename The *.dsa file.
	 * @return void
	 */
	void storeFrames(const std::string& filename);


	/**
	 * Pretty-prints property tree from a certain level onwards.
	 * @param pt The property tree.
	 * @param level The tree level.
	 * @return void
	 */
	void print_tree(const boost::property_tree::ptree& pt, int level);


	/**
	 * Pretty-prints an entire property tree
	 * @param pt The property tree.
	 * @return void
	 */
	void print_tree(const boost::property_tree::ptree& pt);


	/**
	 * Converts a hex string to a float using stringstream.
	 * @param s The hex string.
	 * @return The converted float.
	 */
	float hex2float(std::string& s);


	/**
	 * Converts a float to a hex string.
	 * @param f The converted float.
	 * @return The hex string.
	 */
	std::string float2hex(float f);


	/**
	 * Decode "Enhanced RLE" compressed sensor frame
	 * @details Only zero-valued matrix elements are run length encoded.
	 * 	        Each token t consists of a single precision floating point value (IEEE-754).
	 * 	        t < 0 indicates |t| consecutive zeros.
	 * 	        t > 0: The value of t represents a single observation.
	 * @note Tested with Little-Endian byte order only.
	 * @param hexdata Encoded hexadecimal string.
	 * @param decoded_frame The resulting decoded vector of taxel values.
	 * @return void
	 */
	void decodeTSFrame(const std::string& hexdata, std::vector<float>& decoded_frame);


	/**
	 * Frames are encoded using an "Enhanced RLE Compression"
	 * @details Meaning only zero-valued matrix elements are run length encoded.
	 *          Each token t consists of a single precision floating point value (IEEE-754).
	 *          t < 0 indicates |t| consecutive zeros.
	 *          t > 0: The value of t represents a single observation.
	 * @note Tested with Little-Endian byte order only.
	 * @param cells A vector of taxel values.
	 * @param hexdata The hexadecimal representation.
	 * @return void
	 */
	void encodeTSFrame(const std::vector<float>& cells, std::string& hexdata);


	/**
	 * Loads a SDH-2 pressure profile stored in *.dsa files
	 * @details The file format is xml based and inspired by Weiss's dsa3 format.
	 *          By removing the additional temperature and joint angle readings from the xml tree
	 *          the profiles can be opened in the DSA Explorer by Weiss.
	 * @param filename The filename.
	 * @return void
	 */
	void loadProfile(const std::string& filename);

	/**
	 * Store SDH-2 pressure profile in *.dsa file
	 * @details The file format is xml based and inspired by Weiss's dsa3 format.
	 *          By removing the additional temperature and joint angle readings from the xml tree
	 *          the profiles can be opened in the DSA Explorer by Weiss.
	 * @param filename The filename.
	 * @return void
	 */
	void storeProfile(const std::string& filename);


	/**
	 * Only stores a selection of tactile sensor frames and corresponding temperature and joint angles (including both limits).
	 * See storeProfile() for comparison.
	 * @param filename The filename.
	 * @param fromIdxTS Frame ID from.
	 * @param toIdxTS Frame ID to.
	 * @return void
	 */
	void storeProfileSelection(const string& filename, uint fromIdxTS, uint toIdxTS);


	/**
	 * Pretty-printing of joint angle readings.
	 * @param jointAngleFrame The joint angle frame.
	 * @return void
	 */
	void printJointAngleFrame(JointAngleFrame& jointAngleFrame);


	/**
	 * Pretty-printing of temperature readings.
	 * @param temperatureFrame The temperature frame.
	 * @return void
	 */
	void printTemperatureFrame(TemperatureFrame& temperatureFrame);


	/**
	 * Pretty-printing of a single tactile sensor matrix.
	 * @param frameID The temperature frame.
	 * @param m The matrix ID.
	 * @return void
	 */
	void printTSMatrix(uint frameID, uint m);


	/**
	 * Pretty-printing of all tactile sensor matrices.
	 * @param frameID
	 * @return void
	 */
	void printTSMatrices(uint frameID);

};

#endif /* FRAMEMANAGER_HPP_ */
