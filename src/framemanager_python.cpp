#include <algorithm>

#include <boost/numpy.hpp>
#include <boost/scoped_array.hpp>

#include "NumPyArrayData.h" // Helper for easy array access

#include "framemanager.h"
#include "featureExtraction.h"

namespace bp = boost::python;
namespace np = boost::numpy;

// Macros for fast, direct memory access to NumPy array data
#define NDARRAY_GetPtr2d(dtype,data,strides,i,j) reinterpret_cast<dtype*>(data + i*strides[0] + j*strides[1])

/**
 * @class FrameManagerWrapper
 * @brief Defines Boost.Python wrappers around the FrameManage class.
 *        See Python examples for usage.
 */
class FrameManagerWrapper {

private:
    FrameManager frameManager;
    FrameProcessor *frameProcessor;

public:

    FrameManagerWrapper() {
        frameProcessor = frameManager.getFrameProcessor();
    }

    FrameManagerWrapper(std::string filename) {
        frameProcessor = frameManager.getFrameProcessor();
        frameManager.loadProfile(filename);
    }

    FrameManager& get_framemanager() { 
        return frameManager; 
    }

    void load_profile(std::string filename) {
        frameManager.loadFrames(filename);
    }

    uint get_tsframe_count() {
        return frameManager.getFrameCountTS();
    }

    uint64_t get_tsframe_timestamp(int frameID) {
        return frameManager.getFrame(frameID)->timestamp;
    }

    np::ndarray get_tsframe_timestamp_list() {
        np::ndarray array = np::zeros(bp::make_tuple(frameManager.getFrameCountTS()), np::dtype::get_builtin<uint64_t>());
        for(uint frameID = 0; frameID < frameManager.getFrameCountTS(); frameID++) {
            array[frameID] = frameManager.getFrame(frameID)->timestamp;;
        }
        return array;
    }

    np::ndarray get_tsframe(int frameID, int matrixID) {

        matrixInfo &matrixInfo = frameManager.getMatrixInfo(matrixID);
        int rows = matrixInfo.cells_y;
        int cols = matrixInfo.cells_x;

        TSFrame* tsFrame = frameManager.getFrame(frameID);
        float *data = tsFrame->cells.data();
        data += matrixInfo.texel_offset;

        // Prepare numpy array
        np::dtype dtype = np::dtype::get_builtin<float>();
        int dtype_size = dtype.get_itemsize();

        // Shape
        bp::tuple shape = bp::make_tuple(rows, cols);

        // Stride in bytes per row, bytes per column element
        bp::tuple stride = bp::make_tuple(cols*dtype_size, dtype_size); 

        // Owner, to keep track of memory
        bp::object owner;

        return np::from_data(data, dtype, shape, stride, owner);
    }

    bp::list get_tsframe_list(int frameID) {
        bp::list frameList;
        for(uint matrixID = 0; matrixID < frameManager.getNumMatrices(); matrixID++) {
            np::ndarray matrix_numpy = get_tsframe(frameID, matrixID);
            frameList.append(matrix_numpy);    
        }
        return frameList;
    }


    void set_filter_none() {
        frameManager.setFilterNone();
    }

    void set_filter_median(int kernel_radius, bool masked) {
        frameManager.setFilterMedian(kernel_radius, masked);
    }

    void set_filter_gaussian(int kernel_radius, double sigma) {
        frameManager.setFilterGaussian(kernel_radius, sigma, cv::BORDER_REPLICATE);
    }

    void set_filter_bilateral(int kernel_radius, double sigma_color, double sigma_space) {
        frameManager.setFilterBilateral(kernel_radius, sigma_color, sigma_space, cv::BORDER_REPLICATE);
    }

    void set_filter_morphological(int kernel_type, int kernel_radius, bool masked) {
        frameManager.setFilterMorphological(kernel_type, kernel_radius, masked, cv::BORDER_REPLICATE);
    }

    np::ndarray get_filtered_tsframe(uint frameID, int matrixID) {
        matrixInfo &matrixInfo = frameManager.getMatrixInfo(matrixID);
        int rows = matrixInfo.cells_y;
        int cols = matrixInfo.cells_x;

        TSFrame* tsFrame = frameManager.getFilteredFrame(frameID);
        float *data = tsFrame->cells.data();
        data += matrixInfo.texel_offset;

        // Prepare numpy array
        np::dtype dtype = np::dtype::get_builtin<float>();
        int dtype_size = dtype.get_itemsize();

        // Shape
        bp::tuple shape = bp::make_tuple(rows, cols);

        // Stride in bytes per row, bytes per column element
        bp::tuple stride = bp::make_tuple(cols*dtype_size, dtype_size); 

        // Owner, to keep track of memory
        bp::object owner;

        return np::from_data(data, dtype, shape, stride, owner);
    }   

    bp::list get_filtered_tsframe_list(int frameID) {
        bp::list frameList;
        for(uint matrixID = 0; matrixID < frameManager.getNumMatrices(); matrixID++) {
            np::ndarray matrix_numpy = get_filtered_tsframe(frameID, matrixID);
            frameList.append(matrix_numpy);    
        }
        return frameList;
    }

    double get_texel(uint frameID, uint matrixID, uint x, uint y) {
        return frameManager.getTexel(frameID, matrixID, x, y);
    }

    np::ndarray get_texel_list(uint matrixID, uint x, uint y) {
        np::ndarray array = np::zeros(bp::make_tuple(frameManager.getFrameCountTS()), np::dtype::get_builtin<double>());
        for(uint frameID = 0; frameID < frameManager.getFrameCountTS(); frameID++) {
            double value = get_texel(frameID, matrixID, x, y);
            array[frameID] = value;
        }
        return array;
    }

    // Characteristic values

    // Average
    double get_average_frame(uint frameID) {
        return frameProcessor->getAverage(frameID);
    }

    np::ndarray get_average_frame_list() {
        np::ndarray array = np::zeros(bp::make_tuple(frameManager.getFrameCountTS()), np::dtype::get_builtin<double>());
        for(uint frameID = 0; frameID < frameManager.getFrameCountTS(); frameID++) {
            double average = get_average_frame(frameID);
            array[frameID] = average;
        }
        return array;
    }

    double get_average_matrix(uint frameID, uint matrixID) {
        return frameProcessor->getMatrixAverage(frameID, matrixID);
    }

    np::ndarray get_average_matrix_list(uint matrixID) {
        np::ndarray array = np::zeros(bp::make_tuple(frameManager.getFrameCountTS()), np::dtype::get_builtin<double>());
        for(uint frameID = 0; frameID < frameManager.getFrameCountTS(); frameID++) {
            double average = get_average_matrix(frameID, matrixID);
            array[frameID] = average;
        }
        return array;
    }

    // Min
    double get_min_frame(uint frameID) {
        return frameProcessor->getMin(frameID);
    }

    np::ndarray get_min_frame_list() {
        np::ndarray array = np::zeros(bp::make_tuple(frameManager.getFrameCountTS()), np::dtype::get_builtin<double>());
        for(uint frameID = 0; frameID < frameManager.getFrameCountTS(); frameID++) {
            double min = get_min_frame(frameID);
            array[frameID] = min;
        }
        return array;
    }

    double get_min_matrix(uint frameID, uint matrixID) {
        return frameProcessor->getMatrixMin(frameID, matrixID);
    }

    np::ndarray get_min_matrix_list(uint matrixID) {
        np::ndarray array = np::zeros(bp::make_tuple(frameManager.getFrameCountTS()), np::dtype::get_builtin<double>());
        for(uint frameID = 0; frameID < frameManager.getFrameCountTS(); frameID++) {
            double min = get_min_matrix(frameID, matrixID);
            array[frameID] = min;
        }
        return array;
    }

    // Max
    double get_max_frame(uint frameID) {
        return frameProcessor->getMax(frameID);
    }

    np::ndarray get_max_frame_list() {
        np::ndarray array = np::zeros(bp::make_tuple(frameManager.getFrameCountTS()), np::dtype::get_builtin<double>());
        for(uint frameID = 0; frameID < frameManager.getFrameCountTS(); frameID++) {
            double max = get_max_frame(frameID);
            array[frameID] = max;
        }
        return array;
    }

    double get_max_matrix(uint frameID, uint matrixID) {
        return frameProcessor->getMatrixMax(frameID, matrixID);
    }

    np::ndarray get_max_matrix_list(uint matrixID) {
        np::ndarray array = np::zeros(bp::make_tuple(frameManager.getFrameCountTS()), np::dtype::get_builtin<double>());
        for(uint frameID = 0; frameID < frameManager.getFrameCountTS(); frameID++) {
            double max = get_max_matrix(frameID, matrixID);
            array[frameID] = max;
        }
        return array;
    }

    // Active cells
    int get_num_active_cells_frame(uint frameID) {
        return frameProcessor->getNumActiveCells(frameID);
    }

    int get_num_active_cells_matrix(uint frameID, uint matrixID) {
        return frameProcessor->getMatrixNumActiveCells(frameID, matrixID);
    }


    //--------------
    // Joint Angles
    //--------------
    int get_jointangle_frame_count() {
        return frameManager.getFrameCountJointAngles(); 
    }

    np::ndarray get_jointangle_frame(int angleID) {
        np::ndarray array = np::zeros(bp::make_tuple(7), np::dtype::get_builtin<double>());
        for(uint i = 0; i < 7; i++) {
            array[i] = frameManager.getJointAngleFrame(angleID)->angles[i];
        }
        return array;
    }

    np::ndarray get_jointangle_frame_list() {
        np::ndarray array = np::zeros(bp::make_tuple(frameManager.getFrameCountJointAngles(), 7), np::dtype::get_builtin<double>());

        NumPyArrayData<double> array_data(array);

        for(uint angleID = 0; angleID < frameManager.getFrameCountJointAngles(); angleID++) {
            for(uint i = 0; i < 7; i++) {
                array_data(angleID,i) = frameManager.getJointAngleFrame(angleID)->angles[i];
            }
        }
        return array;
    }

    uint64_t get_jointangle_frame_timestamp(int angleID) {
        return frameManager.getJointAngleFrame(angleID)->timestamp;
    }

    np::ndarray get_jointangle_frame_timestamp_list() {
        np::ndarray array = np::zeros(bp::make_tuple(frameManager.getFrameCountJointAngles()), np::dtype::get_builtin<uint64_t>());
        for(uint frameID = 0; frameID < frameManager.getFrameCountJointAngles(); frameID++) {
            array[frameID] = frameManager.getJointAngleFrame(frameID)->timestamp;;
        }
        return array;
    }

    //--------------
    // Temperatures
    //--------------
    int get_temperature_frame_count() {
        return frameManager.getFrameCountTemperature(); 
    }

    np::ndarray get_temperature_frame(int tempID) {
        np::ndarray array = np::zeros(bp::make_tuple(9), np::dtype::get_builtin<double>());
        for(uint i = 0; i < 9; i++) {
            array[i] = frameManager.getTemperatureFrame(tempID)->values[i];
        }
        return array;
    }

    np::ndarray get_temperature_frame_list() {
        np::ndarray array = np::zeros(bp::make_tuple(frameManager.getFrameCountTemperature(), 9), np::dtype::get_builtin<double>());

        NumPyArrayData<double> array_data(array);

        for(uint tempID = 0; tempID < frameManager.getFrameCountTemperature(); tempID++) {
            for(uint i = 0; i < 9; i++) {
                array_data(tempID,i) = frameManager.getTemperatureFrame(tempID)->values[i];
            }
        }
        return array;
    }

    uint64_t get_temperature_frame_timestamp(int tempID) {
        return frameManager.getTemperatureFrame(tempID)->timestamp;
    }

    np::ndarray get_temperature_frame_timestamp_list() {
        np::ndarray array = np::zeros(bp::make_tuple(frameManager.getFrameCountTemperature()), np::dtype::get_builtin<uint64_t>());
        for(uint tempID = 0; tempID < frameManager.getFrameCountTemperature(); tempID++) {
            array[tempID] = frameManager.getTemperatureFrame(tempID)->timestamp;
        }
        return array;
    }

    //------------------------------------------------
    // Mapping: TSframe -> Corresponding joint angles
    //------------------------------------------------
    np::ndarray get_corresponding_jointangles(int tsframeID) {
        // Get corresponding angles
        JointAngleFrame *jointAngleFrame = frameManager.getCorrespondingJointAngle(tsframeID);

        np::ndarray array = np::zeros(bp::make_tuple(7), np::dtype::get_builtin<double>());
        for(uint i = 0; i < 7; i++) {
            array[i] = jointAngleFrame->angles[i];
        }
        return array;
    }

    np::ndarray get_corresponding_jointangles_list() {
        np::ndarray array = np::zeros(bp::make_tuple(frameManager.getFrameCountTS(), 7), np::dtype::get_builtin<double>());

        NumPyArrayData<double> array_data(array);

        JointAngleFrame *jointAngleFrame;
        for(uint tsframeID = 0; tsframeID < frameManager.getFrameCountTS(); tsframeID++) {
            // Get corresponding angles
            jointAngleFrame = frameManager.getCorrespondingJointAngle(tsframeID);
            for(uint i = 0; i < 7; i++) {
                array_data(tsframeID, i) = jointAngleFrame->angles[i];
            }
        }
        return array;
    }

    //------------------------------------------------
    // Mapping: TSframe -> Corresponding temperatures
    //-------------------------------------------------
    np::ndarray get_corresponding_temperatures(int tsframeID) {
        // Get corresponding temperatures
        TemperatureFrame *tempFrame = frameManager.getCorrespondingTemperature(tsframeID);

        np::ndarray array = np::zeros(bp::make_tuple(9), np::dtype::get_builtin<double>());
        for(uint i = 0; i < 9; i++) {
            array[i] = tempFrame->values[i];
        }
        return array;
    }

    np::ndarray get_corresponding_temperatures_list() {
        np::ndarray array = np::zeros(bp::make_tuple(frameManager.getFrameCountTS(), 9), np::dtype::get_builtin<double>());

        NumPyArrayData<double> array_data(array);

        for(uint tsframeID = 0; tsframeID < frameManager.getFrameCountTS(); tsframeID++) {
            // Get corresponding temperatures
            TemperatureFrame *tempFrame = frameManager.getCorrespondingTemperature(tsframeID);

            for(uint i = 0; i < 9; i++) {
                array_data(tsframeID, i) = tempFrame->values[i];
            }
        }
        return array;
    }


}; 


/**
 * @class FeatureExtractionWrapper
 * @brief Defines Boost.Python wrappers around the FeatureExtraction class.
 *        See Python examples for usage.
 */
class FeatureExtractionWrapper {

private:
    FrameManager& frameManager;
    FeatureExtraction featureExtraction;

public:

    FeatureExtractionWrapper(FrameManagerWrapper& fmw) 
    : frameManager(fmw.get_framemanager()),
      featureExtraction(fmw.get_framemanager()) { 
    }

    np::ndarray computeCentroid(int frameID, int matrixID) {
        std::vector<double> centroid = featureExtraction.computeCentroid(frameID, matrixID);
        np::ndarray np_centroid = np::zeros(bp::make_tuple(2), np::dtype::get_builtin<double>());
        np_centroid[0] = centroid[0];
        np_centroid[1] = centroid[1];
        return np_centroid;
    }

    double computeStandardDeviation(int frameID, int matrixID)  {
        return featureExtraction.computeStandardDeviation(frameID, matrixID);
    }

    bp::list computeStandardDeviationList(int frameID) {
        bp::list frameList;
        for(uint matrixID = 0; matrixID < frameManager.getNumMatrices(); matrixID++) {
            double stdDev = featureExtraction.computeStandardDeviation(frameID, matrixID);
            frameList.append(stdDev);    
        }
        return frameList;
    }

    np::ndarray computeChebyshevMoments(int frameID, int matrixID, int pmax) {

        array_type T_pq_doubleprime = featureExtraction.computeMoments(frameID, matrixID, pmax);

        // Prepare numpy array
        np::dtype dtype = np::dtype::get_builtin<double>();
        int dtype_size = dtype.get_itemsize();

        // Shape
        bp::tuple shape = bp::make_tuple(pmax, pmax);

        // Stride in bytes per row, bytes per column element
        bp::tuple stride = bp::make_tuple(pmax*dtype_size, dtype_size); 

        // Owner, to keep track of memory
        bp::object owner;

        np::ndarray matrix_numpy = np::from_data(T_pq_doubleprime.data(), dtype, shape, stride, owner);

        return matrix_numpy.copy();
    }

    bp::list computeChebyshevMomentsList(int frameID, int pmax) {
        bp::list frameList;
        for(uint matrixID = 0; matrixID < frameManager.getNumMatrices(); matrixID++) {
            np::ndarray matrix_numpy = computeChebyshevMoments(frameID, matrixID, pmax);
            frameList.append(matrix_numpy);    
        }
        return frameList;
    }

    // Returns center (x,y,z) and radius
    np::ndarray computeMinimalBoundingSphere(int frameID, np::ndarray& phi_ndarray)  {
        NumPyArrayData<double> phi_data(phi_ndarray);
        std::vector<double> values = featureExtraction.computeMiniball(frameID, phi_data(0), phi_data(1), phi_data(2), phi_data(3), phi_data(4), phi_data(5), phi_data(6));

        uint length = values.size();
        np::ndarray np_values = np::zeros(bp::make_tuple(length), np::dtype::get_builtin<double>());
        for(uint i = 0; i < length; i++) {
            np_values[i] = values[i];
        }

        return np_values;
    }

    // Returns center (x,y,z) and radius
    np::ndarray computeMinimalBoundingSphereCentroid(int frameID, np::ndarray& phi_ndarray)  {
        NumPyArrayData<double> phi_data(phi_ndarray);
        std::vector<double> values = featureExtraction.computeMiniballCentroid(frameID, phi_data(0), phi_data(1), phi_data(2), phi_data(3), phi_data(4), phi_data(5), phi_data(6));

        uint length = values.size();
        np::ndarray np_values = np::zeros(bp::make_tuple(length), np::dtype::get_builtin<double>());
        for(uint i = 0; i < length; i++) {
            np_values[i] = values[i];
        }

        return np_values;
    }

    // Returns center (x,y,z) and radius
    np::ndarray computeMinimalBoundingSpherePoints(np::ndarray& taxel_ndarray, np::ndarray& phi_ndarray)  {

        // Copy numpy array of points to 2D vector
        NumPyArrayData<double> taxel_data(taxel_ndarray);
        std::vector< std::vector<double> > taxels_vec(taxel_ndarray.shape(0), std::vector<double>(3));
        for (int i = 0; i < taxel_ndarray.shape(0); i++) {
            for (int j=0; j < taxel_ndarray.shape(1); j++) {   
                taxels_vec[i][j] = taxel_data(i,j);
            }
        }

        // Copy numpy array of joint angles to std::vector
        NumPyArrayData<double> phi_data(phi_ndarray);
        std::vector<double> angles(7);
        for (int i = 0; i < 7; i++) {
            angles[i] = phi_data(i);
        }

        // Compute Miniball
        std::vector<double> values = featureExtraction.computeMiniballPoints(taxels_vec, angles);

        uint length = values.size();
        np::ndarray np_values = np::zeros(bp::make_tuple(length), np::dtype::get_builtin<double>());
        for(uint i = 0; i < length; i++) {
            np_values[i] = values[i];
        }

        return np_values;
    }

};


BOOST_PYTHON_MODULE(framemanager_python)
{
    np::initialize();  // have to put this in any module that uses Boost.NumPy
    
    bp::class_<FrameManagerWrapper, boost::noncopyable>("FrameManagerWrapper")  // Default constructor, Note: noncopyable because of boost::mutex and boost:: condition_variable
    .def(bp::init<std::string>() ) // Alternative constructor
    .def("get_framemanager", &FrameManagerWrapper::get_framemanager, bp::return_internal_reference<>()) // Returns an internal reference
    .def("load_profile", &FrameManagerWrapper::load_profile)
    .def("get_tsframe_count", &FrameManagerWrapper::get_tsframe_count)
    .def("get_tsframe", &FrameManagerWrapper::get_tsframe)
    .def("get_tsframe_list", &FrameManagerWrapper::get_tsframe_list)
    .def("get_filtered_tsframe", &FrameManagerWrapper::get_filtered_tsframe)
    .def("get_filtered_tsframe_list", &FrameManagerWrapper::get_filtered_tsframe_list)
    .def("get_tsframe_timestamp", &FrameManagerWrapper::get_tsframe_timestamp)
    .def("get_tsframe_timestamp_list", &FrameManagerWrapper::get_tsframe_timestamp_list)
    
    // OpenCV filter
    .def("set_filter_none", &FrameManagerWrapper::set_filter_none)
    .def("set_filter_median", &FrameManagerWrapper::set_filter_median)
    .def("set_filter_gaussian", &FrameManagerWrapper::set_filter_gaussian)
    .def("set_filter_bilateral", &FrameManagerWrapper::set_filter_bilateral)
    .def("set_filter_morphological", &FrameManagerWrapper::set_filter_morphological)
    
    // Single cell
    .def("get_texel", &FrameManagerWrapper::get_texel)
    .def("get_texel_list", &FrameManagerWrapper::get_texel_list)
    
    // Average
    .def("get_average_matrix", &FrameManagerWrapper::get_average_matrix)
    .def("get_average_frame", &FrameManagerWrapper::get_average_frame)
    .def("get_average_matrix_list", &FrameManagerWrapper::get_average_matrix_list)
    .def("get_average_frame_list", &FrameManagerWrapper::get_average_frame_list)
    
    // Min
    .def("get_min_matrix", &FrameManagerWrapper::get_min_matrix)
    .def("get_min_frame", &FrameManagerWrapper::get_min_frame)
    .def("get_min_matrix_list", &FrameManagerWrapper::get_min_matrix_list)
    .def("get_min_frame_list", &FrameManagerWrapper::get_min_frame_list)

    // Max
    .def("get_max_matrix", &FrameManagerWrapper::get_max_matrix)
    .def("get_max_frame", &FrameManagerWrapper::get_max_frame)
    .def("get_max_matrix_list", &FrameManagerWrapper::get_max_matrix_list)
    .def("get_max_frame_list", &FrameManagerWrapper::get_max_frame_list)
    
    .def("get_num_active_cells_matrix", &FrameManagerWrapper::get_num_active_cells_matrix)
    .def("get_num_active_cells_frame", &FrameManagerWrapper::get_num_active_cells_frame)
      
    .def("get_jointangle_frame_count", &FrameManagerWrapper::get_jointangle_frame_count)
    .def("get_jointangle_frame", &FrameManagerWrapper::get_jointangle_frame)
    .def("get_jointangle_frame_list", &FrameManagerWrapper::get_jointangle_frame_list)
    .def("get_jointangle_frame_timestamp", &FrameManagerWrapper::get_jointangle_frame_timestamp)
    .def("get_jointangle_frame_timestamp_list", &FrameManagerWrapper::get_jointangle_frame_timestamp_list)
    
    .def("get_temperature_frame_count", &FrameManagerWrapper::get_temperature_frame_count)
    .def("get_temperature_frame", &FrameManagerWrapper::get_temperature_frame)
    .def("get_temperature_frame_list", &FrameManagerWrapper::get_temperature_frame_list)
    .def("get_temperature_frame_timestamp", &FrameManagerWrapper::get_temperature_frame_timestamp)
    .def("get_temperature_frame_timestamp_list", &FrameManagerWrapper::get_temperature_frame_timestamp_list)
    
    .def("get_corresponding_jointangles", &FrameManagerWrapper::get_corresponding_jointangles)
    .def("get_corresponding_jointangles_list", &FrameManagerWrapper::get_corresponding_jointangles_list)
    .def("get_corresponding_temperatures", &FrameManagerWrapper::get_corresponding_temperatures)
    .def("get_corresponding_temperatures_list", &FrameManagerWrapper::get_corresponding_temperatures_list)
    ;
    
    bp::class_<FeatureExtractionWrapper>("FeatureExtractionWrapper", bp::init<FrameManagerWrapper &>())  // Constructor
    .def("compute_centroid", &FeatureExtractionWrapper::computeCentroid)
    .def("compute_standard_deviation", &FeatureExtractionWrapper::computeStandardDeviation)
    .def("compute_standard_deviation_list", &FeatureExtractionWrapper::computeStandardDeviationList)
    .def("compute_chebyshev_moments", &FeatureExtractionWrapper::computeChebyshevMoments)
    .def("compute_chebyshev_moments_list", &FeatureExtractionWrapper::computeChebyshevMomentsList)
    .def("compute_minimal_bounding_sphere", &FeatureExtractionWrapper::computeMinimalBoundingSphere)
    .def("compute_minimal_bounding_sphere_centroid", &FeatureExtractionWrapper::computeMinimalBoundingSphereCentroid)
    .def("compute_minimal_bounding_sphere_points", &FeatureExtractionWrapper::computeMinimalBoundingSpherePoints)
    ;
}
