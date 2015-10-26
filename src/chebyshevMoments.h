#ifndef CHEBYSHEV_MOMENTS_H_
#define CHEBYSHEV_MOMENTS_H_

#include <boost/multi_array.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/binomial.hpp>

/// @typedef Boost Multiarray type for 2D moments
typedef boost::multi_array<double, 2> array_type;

/**
 * @class Chebyshev
 * @brief Computes discrete Chebyshev polynomaials and thereon based image moments (translation- and rotation invariant).
 * @note See Chapter 6.6, Chebyshev moments of my thesis for used formulas and further details.
 *       Based on publications by Mukundan et al. as well as "Moments and Moment Invariants in Pattern Recognition", Jan Flusser, Barbara Zitova and Tomas Suk.
 *       The look-up table approach is following: "Symmetric image recognition by Tchebichef moment invariants", Hui Zhang, Xiubing Dai, Pei Sun, Hongqing Zhu, and Huazhong Shu.
 */
class Chebyshev {

private:
	int m_pmax_rot; // Maximum rotation invariant moment order (user defined)
	int m_pmax; // Resulting internal moment order
	int m_rows; // Frame height
	int m_cols; // Frame width
	int m_N; // Size of final padded frame

	// Border padding
	int m_top;
	int m_bottom;
	int m_left;
    int m_right;

    // Internal copies for optional image reconstruction
	array_type m_C;
	array_type m_D;
	array_type m_t_p;
	array_type m_T_pq;

	/**
	 * Computes the squared norm of the polynomial set (orthonormal).
	 * @param[in] pmax Maximum moment order.
	 * @param[in] N Image size.
	 * @param[out] beta Reference to the resulting norm.
	 * @return void
	 */
	void computeNorm(int pmax, int N, std::vector<double> &beta);

	/**
	 * Computes a table of signed Stirling numbers of the first kind up to \a N.
	 * @param[in] N Number of Stirling numbers.
	 * @param[out] S1 Reference to the table containing the Stirling numbers.
	 * @return void
	 */
	void computeS1(int N, array_type& S1);

	/**
	 * Computes a table of signed Stirling numbers of the second kind up to \a N.
	 * @param[in] N Number of Stirling numbers.
	 * @param[out] S2 Reference to the table containing the Stirling numbers.
	 * @return void
	 */
	void computeS2(int N, array_type& S2);

	/**
	 * Computes lower triangular matrix \b C.
	 * @param[in] pmax Maximum moment order.
	 * @param[in] N Image size.
	 * @param[in] beta Reference to the squared norm.
	 * @param[in] S1 Reference to the precomputed Stirling numbers.
	 * @param[out] C Reference to the resulting lower triangular matrix \b C.
	 * @return void
	 */
	void computeMatrixC(int pmax, int N, std::vector<double>& beta, array_type& S1, array_type& C);

	/**
	 * Computes upper triangular matrix \b D by inverting \C.
	 * @param[in] pmax Maximum moment order.
	 * @param[in] N Image size.
 	 * @param[in] C Reference to the lower triangular matrix \b C.
	 * @param[out] D Reference to the resulting triangular matrix \b D.
	 * @return void
	 */
	void computeMatrixD(int pmax, int N, array_type& C, array_type& D);

	/**
	 * Computes values of orthonormal Chebyshev polynomials at the supporting points.
	 * @param[in] pmax Maximum moment order.
	 * @param[in] N Image size.
 	 * @param[in] C Reference to the lower triangular matrix \b C.
	 * @param[out] t_p Reference to the resulting values.
	 * @return void
	 */
	void computePolynomials(int pmax, int N, array_type& C, array_type& t_p);

	/**
	 * Compute basic Chebyshev moments
	 * @param[in] pmax Maximum moment order.
	 * @param[in] N Image size.
 	 * @param[in] t_p Reference to the values of the Chebyshev polynomials at the supporting points.
 	 * @param[in] frame Reference to the tactile image.
  	 * @param[out] T_pq Reference to the resulting ordinary Chebyshev moments.
	 */
	void computeMoments(int pmax, int N, array_type& t_p, cv::Mat& frame, array_type& T_pq);

	/**
	 * Compute translation invariant Chebyshev moments
	 * @param[in] pmax Maximum moment order.
	 * @param[in] N Image size.
 	 * @param[in] C Reference to the lower triangular matrix \b C.
 	 * @param[in] D Reference to the upper triangular matrix \b D.
 	 * @param[in] centroid_x, centroid_y Centroid, i.e. Center of gravity of ordinary Chebyshev moments
  	 * @param[in] T_pq Reference to ordinary Chebyshev moments.
  	 * @param[out] T_pq_prime Reference to the resulting translation invariant Chebyshev moments.
	 */
	void computeMoments_prime(int pmax, int N, array_type& C, array_type& D, double centroid_x, double centroid_y, array_type& T_pq, array_type& T_pq_prime);

	/**
	 * Compute translation and rotation invariant Chebyshev moments
	 * @param[in] pmax Maximum moment order.
	 * @param[in] N Image size.
 	 * @param[in] C Reference to the lower triangular matrix \b C.
 	 * @param[in] D Reference to the upper triangular matrix \b D.
  	 * @param[in] T_pq_prime Reference to translation invariant Chebyshev moments.
  	 * @param[out] T_pq_doubleprime Reference to the final translation and rotation invariant Chebyshev moments.
	 */
	void computeMoments_doubleprime(int pmax, int N, array_type& C, array_type& D, array_type& T_pq_prime, array_type& T_pq_doubleprime);

	/**
	 * Precompute look-up tables for given moment order and frame size
	 * @param[in] pmax Maximum moment order.
	 * @param[in] N Image size.
	 * @param[in] t_p Reference to the resulting values.
  	 * @param[in] T_pq Reference to ordinary Chebyshev moments.
	 * @param[out] frame Reference to the resulting tactile image.
	 */
	void computeReconstruction(int pmax, int N, array_type& t_p, array_type& T_pq, cv::Mat& frame);

public:

	/**
	 * Constructor
	 */
	Chebyshev() { m_pmax_rot = 0; m_N = 0; };

	/**
	 * Initializes look-up tables that stay the same as long as the image size and moment order does not change.
	 * @note In order to compute rotation invariant moments of order p, "normal" moments of order 2*(pmax_rot-1)+1 are needed.
	 * @param[in] frame Reference to the tactile image.
	 * @param[in] pmax_rot Maximum rotation invariant moment order.
	 * @return void
	 */
	void initialize(cv::Mat& frame, int pmax_rot);

	/**
	 * Computes discrete translation and rotation invariant Chebyshev moments
	 * @note In order to compute rotation invariant moments of order p, "normal" moments of order 2*(pmax_rot-1)+1 are needed.
	 * @param[in] frame Reference to the tactile image.
	 * @param[in] pmax Maximum invariant moment order.
	 * @param[out] T_pq_doubleprime Reference to final rotation and translation invariant Chebyshev moments.
	 * @return void
	 */
	void computeInvariants(cv::Mat& frame, int pmax, array_type& T_pq_doubleprime);

	/**
	 * Performs a reconstruction from moments assuming computeInvariants() has been executed before.
	 * @param[out] frame Reference to the reconstructed image.
	 * @return void
	 */
	void computeReconstruction(cv::Mat& frame);
};

#endif /* CHEBYSHEV_MOMENTS_H_ */
