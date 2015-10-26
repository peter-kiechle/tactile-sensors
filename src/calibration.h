#ifndef CALIBRATION_H_
#define CALIBRATION_H_

#include <vector>
#include <string>

/// A data structure containing the linear regression parameters as well as the RMS Error of the prediction band
struct TemperatureNoise {
	double slope;
	double intercept;
	double RMSE;
};

/**
 * @class Calibration
 * @brief Reads an XML file containing the calibration parameters for the High-Sensitivity Mode
 * @note  Regression parameters were determined with "noise-temperature_calibration.py"
 *        Values could be tweaked manually if wear and tear changes the sensor's behavior
 */
class Calibration {
public:

    /**
     * Constructor that immediately imports the file "calibration_temperature_noise.xml"
     */
	Calibration();
	virtual ~Calibration();

    /**
     * Reads calibration parameter file
	 *
     * @param filename The XML file containing the calibrated parameters
     * @return void
     */
	void readTemperatureNoise(const std::string& filename);

    /**
     * Returns the calibrated slope, intercept and RMSE of the specified sensor matrix
     *
     * @param matrixID Specified sensor matrix
     * @return The linear regression parameters
     */
	TemperatureNoise& getTemperatureNoise(uint matrixID);

private:
	std::vector<TemperatureNoise> parameters;

};

#endif /* CALIBRATION_H_ */
