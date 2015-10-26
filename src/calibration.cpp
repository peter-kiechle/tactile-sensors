#include <fstream>
#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "calibration.h"

Calibration::Calibration() {
	readTemperatureNoise("calibration_temperature_noise.xml");
}

Calibration::~Calibration() {
}

void Calibration::readTemperatureNoise(const std::string& filename) {

	std::ifstream infile(filename.c_str(), std::ios_base::in | std::ios_base::binary);
	if (!infile) {
		std::cerr<< "Error! Can't open file: " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	using namespace boost::property_tree;

	// Read XML file
	ptree xmlTree;
	try {
		read_xml(infile, xmlTree, xml_parser::trim_whitespace);
	}
	catch (std::exception &e) {
		std::string what = e.what();
		std::cerr << "Error while parsing file: " << what << std::endl;
	}

	// Traverse tree
	parameters.clear();
	parameters.resize(6);
	BOOST_FOREACH(ptree::value_type const&v, xmlTree.get_child("calibration")) {
		if(v.first == "matrix") {
			const ptree& node = v.second;
			int id = node.get<int>("<xmlattr>.id");
			TemperatureNoise param;
			param.slope = node.get<double>("slope");
			param.intercept = node.get<double>("intercept");
			param.RMSE = node.get<double>("RMSE");
			parameters.insert(parameters.begin() + id, param);
		}
	}
}

TemperatureNoise& Calibration::getTemperatureNoise(uint matrixID) {
	return parameters.at(matrixID);
}
