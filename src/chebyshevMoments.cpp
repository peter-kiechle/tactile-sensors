#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "chebyshevMoments.h"

void Chebyshev::computeNorm(int pmax, int N, std::vector<double> &beta) {
	using namespace boost::math;
	double numerator, denominator;

	// Iterative version (thanks to tail recursion)
	for(int p = 0; p < pmax; p++) {
		numerator = factorial<double>(N+p);
		denominator = (2*p+1) * factorial<double>(N-p-1);
		beta[p] = sqrt(numerator / denominator);
	}
}

void Chebyshev::computeS1(int N, array_type& S1) {

	// Initial conditions
	for(int n = 0; n < N; n++) {
		for(int k = 0; k < N; k++) {
			S1[n][k] = 0.0;
		}
	}
	S1[0][0] = 1.0;

	// Dynamic programming
	for (int n = 1; n < N; n++) {
		for (int k = 1; k <= n; k++) {
			//S1[n][k] = S1[n-1][k-1] + (n-1) * S1[n-1][k]; // unsigned version
			S1[n][k] = S1[n-1][k-1] - (n-1) * S1[n-1][k]; // signed version
		}
	}
}

void Chebyshev::computeS2(int N, array_type& S2) {

	// Initial conditions
	for(int n = 0; n < N; n++) {
		for(int k = 0; k < N; k++) {
			S2[n][k] = 0.0;
		}
	}
	S2[0][0] = 1.0;

	for (int n = 1; n < N; n++) {
		for (int k = 1; k <= n; k++) {
			S2[n][k] = S2[n-1][k-1] + k*S2[n-1][k];
		}
	}
}

void Chebyshev::computeMatrixC(int pmax, int N, std::vector<double>& beta, array_type& S1, array_type& C) {
	using namespace boost::math;
	double sum = 0.0;
	double numerator, denominator, sign;

	for(int p = 0; p < pmax; p++) {
		for(int k = 0; k < pmax; k++) {
			sum = 0.0;
			for(int r = k; r <= p; r++) {
				sign = ( (p+r)%2 == 0 ) ? 1.0 : -1.0;
				numerator = sign * factorial<double>(p+r) * factorial<double>(N-r-1);
				denominator = factorial<double>(p-r) * pow(factorial<double>(r), 2) * factorial<double>(N-p-1);
				sum += S1[r][k] * (numerator / denominator);
			}
			C[p][k] = (1.0/beta[p]) * sum;
		}
	}
}

void Chebyshev::computeMatrixD(int pmax, int N, array_type& C, array_type& D) {

	// The paper's closed form formula seems to be broken!!!
	// Solution: simply invert C

	// Ugly conversion, but OpenCV is already linked and does the job ;-)
	cv::Mat D_opencv = cv::Mat(pmax, pmax, CV_64FC1);
	for(int p = 0; p < pmax; p++) {
		for(int k = 0; k < pmax; k++) {
			D_opencv.at<double>(p,k) = C[p][k];
		}
	}

	// Invert C
	cv::invert(D_opencv, D_opencv, cv::DECOMP_LU);

	// Ugly back conversion...
	for(int p = 0; p < pmax; p++) {
		for(int k = 0; k < pmax; k++) {
			D[p][k] = D_opencv.at<double>(p,k);
		}
	}
}

void Chebyshev::computePolynomials(int pmax, int N, array_type& C, array_type& t_p) {
	double sum;
	for(int p = 0; p < pmax; p++) {
		for(int x = 0; x < N; x++) {
			sum = 0.0;
			for(int k = 0; k <= p; k++) {
				sum += C[p][k] * pow(x, k);
			}
			t_p[p][x] = sum;
		}
	}
}

void Chebyshev::computeMoments(int pmax, int N, array_type& t_p, cv::Mat& frame, array_type& T_pq) {
	double sum;
	for(int p = 0; p < pmax; p++) {
		for(int q = 0; q < pmax; q++) {
			sum = 0.0;
			for(int y = 0; y < N; y++) {
				for(int x = 0; x < N; x++) {
					sum += t_p[p][x] * t_p[q][y] * frame.at<float>(y,x);
				}
			}
			T_pq[p][q] = sum;
		}
	}
}

void Chebyshev::computeMoments_prime(int pmax, int N, array_type& C, array_type& D, double c_x, double c_y, array_type& T_pq, array_type& T_pq_prime) {
	using namespace boost::math;
	double binomial_m_s, binomial_n_t;
	double sum;

	for(int p = 0; p < pmax; p++) {
		for(int q = 0; q < pmax; q++) {
			sum = 0.0;
			for(int m = 0; m <= p; m++) {
				for(int n = 0; n <= q; n++) {
					for(int s = 0; s <= m; s++) {
						binomial_m_s = binomial_coefficient<double>(m, s);
						for(int t = 0; t <= n; t++) {
							binomial_n_t = binomial_coefficient<double>(n, t);
							for(int i = 0; i <= s; i++) {
								for(int j = 0; j <= t; j++) {
									sum += binomial_m_s * binomial_n_t * C[p][m] * C[q][n] * D[s][i] * D[t][j] * pow(-c_x, m-s) * pow(-c_y, n-t) * T_pq[i][j];
								}
							}
						}
					}
				}
			}
			T_pq_prime[p][q] = sum;
		}
	}
}

void Chebyshev::computeMoments_doubleprime(int pmax, int N, array_type& C, array_type& D, array_type& T_pq_prime, array_type& T_pq_doubleprime) {
	using namespace boost::math;
	double binomial_m_s, binomial_n_t, sign;
	double sum;
	double u = ( 2.0 * C[2][2] * C[0][0]) / pow(C[1][1],2);
	double v = ( 2.0 * C[2][2] * pow(C[1][0],2) ) / ( C[0][0] * pow(C[1][1],2) );
	double theta = 0.5 * atan( (u*T_pq_prime[1][1] - v*T_pq_prime[0][0]) / (T_pq_prime[2][0] - T_pq_prime[0][2]) );
	double cos_theta = cos(theta);
	double sin_theta = sin(theta);

	// Yes, you're right. 8 nested loops ;-)
	for(int p = 0; p < pmax; p++) {
		for(int q = 0; q < pmax; q++) {
			sum = 0.0;
			for(int m = 0; m <= p; m++) {
				for(int n = 0; n <= q; n++) {
					for(int s = 0; s <= m; s++) {
						binomial_m_s = binomial_coefficient<double>(m, s);
						for(int t = 0; t <= n; t++) {
							binomial_n_t = binomial_coefficient<double>(n, t);
							//sign = ( t%2 == 0 ) ? 1.0 : -1.0;
							sign = pow(-1.0, t);
							for(int i = 0; i <= s+t; i++) {
								for(int j = 0; j <= m+n-s-t; j++) {
									sum += 	  binomial_m_s
											* binomial_n_t
											* sign
											* pow(cos_theta, n+s-t)
											* pow(sin_theta, m+t-s)
											* C[p][m]
											* C[q][n]
											* D[s+t][i]
											* D[m+n-s-t][j]
										    * T_pq_prime[i][j];
								}
							}
						}
					}
				}
			}
			T_pq_doubleprime[p][q] = sum;
		}
	}
}

void Chebyshev::computeReconstruction(cv::Mat& frame) {
	double sum;
	for(int y = 0; y < m_N; y++) {
		for(int x = 0; x < m_N; x++) {
			sum = 0.0;
			for(int p = 0; p < m_pmax; p++) {
				for(int q = 0; q < m_pmax; q++) {
					sum += m_T_pq[p][q] * m_t_p[p][x] * m_t_p[q][y];
				}
			}
			frame.at<float>(y,x) = sum;
		}
	}
}

void Chebyshev::initialize(cv::Mat& frame, int pmax_rot) {

	printf("Initializing Chebyshev Moments: w=%d, h=%d, p=%d\n", frame.cols, frame.rows, pmax_rot);

	// In order to compute rotation invariant moments of order p, "normal" moments of order 2*(pmax_rot-1)+1 are needed
	m_pmax_rot = pmax_rot;
	m_pmax = 2*(m_pmax_rot-1)+1;

	m_N = frame.rows; // Frame should be square by now

	// Compute squared norm (orthonormal)
	std::vector<double> beta(m_pmax);
	computeNorm(m_pmax, m_N, beta);

	// Compute Stirling numbers
	array_type S1(boost::extents[m_N][m_N]);
	computeS1(m_N, S1);

	// Compute lower triangular matrix C
	m_C.resize(boost::extents[m_pmax][m_pmax]);
	computeMatrixC(m_pmax, m_N, beta, S1, m_C);

	// Compute lower triangular matrix D
	m_D.resize(boost::extents[m_pmax][m_pmax]);
	computeMatrixD(m_pmax, m_N, m_C, m_D);

	// Compute orthonormal Chebyshev polynomials
	m_t_p.resize(boost::extents[m_pmax][m_N]);
	computePolynomials(m_pmax, m_N, m_C, m_t_p);
}

void Chebyshev::computeInvariants(cv::Mat& frame, int pmax_rot, array_type& T_pq_doubleprime) {

	// Add padding (square and even)
	m_rows = frame.rows;
	m_cols = frame.cols;
	m_top = 0;
	m_bottom = (m_rows % 2 == 0) ? 0 : 1; // Pad fingertip frames to even size
	m_left = (int)((m_rows+m_bottom - m_cols) / 2.0);
    m_right = (m_rows+m_bottom) - (m_cols+m_left);
    cv::copyMakeBorder(frame, frame, m_top, m_bottom, m_left, m_right, cv::BORDER_CONSTANT, cv::Scalar(0.0, 0.0, 0.0) );

	// Check if look-up tables have already been computed for given moment order and/or frame size
	if(m_pmax_rot != pmax_rot || m_N != frame.rows) {
		initialize(frame, pmax_rot);
	}

	// Compute Chebyshev moments
	array_type T_pq(boost::extents[m_pmax][m_pmax]);
	computeMoments(m_pmax, m_N, m_t_p, frame, T_pq);

	// Store result for possible reconstruction
	m_T_pq.resize(boost::extents[m_pmax][m_pmax]);
	m_T_pq = T_pq;

	// Compute centroids (including padding)
	double centroid_x = (m_C[0][0]*T_pq[1][0] - m_C[1][0]*T_pq[0][0]) / (m_C[1][1] * T_pq[0][0]);
	double centroid_y = (m_C[0][0]*T_pq[0][1] - m_C[1][0]*T_pq[0][0]) / (m_C[1][1] * T_pq[0][0]);

	// Compute translation invariant Chebyshev moments
	array_type T_pq_prime(boost::extents[m_pmax][m_pmax]);
	computeMoments_prime(m_pmax, m_N, m_C, m_D, centroid_x, centroid_y, T_pq, T_pq_prime);

	// Compute translation and rotation invariant Chebyshev moments
	computeMoments_doubleprime(pmax_rot, m_N, m_C, m_D, T_pq_prime, T_pq_doubleprime);
}
