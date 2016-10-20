#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdbool.h>
#include <math.h>

typedef struct point2d {
	double x;
	double y;
} point2d_t;

double dotProduct2d(point2d_t x, point2d_t y);
double fun(point2d_t x);
point2d_t grad_fun(point2d_t x);
double findT_phi(point2d_t x, point2d_t grad_fun_x);
point2d_t nonlinearConjugateGradientMethod(point2d_t x0, double epsilon, uint32_t maxIter);

int32_t main() {
	const point2d_t x0     = { 0.0, 0.0 };
	const double epsilon   = 1.0e-2;
	const uint32_t maxIter = 100;
	const point2d_t xmin   = nonlinearConjugateGradientMethod(x0, epsilon, maxIter);

	printf("\nXmin = (%.2f, %.2f)\n", xmin.x, xmin.y);

	system("pause");

	return 0;
}

inline double dotProduct2d(const point2d_t x, const point2d_t y) {
	return x.x * y.x + x.y * y.y;
}

inline double fun(const point2d_t x) {
	return (x.x - 2.0 * x.y) * (x.x - 2.0 * x.y) + (x.x + 5.0) * (x.x + 5.0);
}

inline point2d_t grad_fun(const point2d_t x) {
	point2d_t grad = { 4.0 * x.x + 4.0 * x.y + 10.0,
					   4.0 * x.x - 8.0 * x.y         };
	return grad;
}

inline double findT_phi(const point2d_t x, const point2d_t d) {
	// Quadratic function
	//if (dotProduct2d(d, d) < 0) {
	//	printf("Error! d^phi/dt^2 < 0\n");
	//	return 0;
	//}
	return ((2.0 * x.y - x.x) * (d.x - 2.0 * d.y) - (5.0 + x.x) * d.x) / dotProduct2d(d, d);
}

point2d_t nonlinearConjugateGradientMethod(const point2d_t x0, const double epsilon, const uint32_t maxIter) {
	const double epsilon2 = epsilon * epsilon;
	point2d_t xk1 = x0;
	point2d_t xk, d;
	bool is_seq = false;
	uint32_t k = 0;

	for (k; k < maxIter; ++k) {
		const point2d_t grad_fun_xk1 = grad_fun(xk1);
		const double norm_grad = dotProduct2d(grad_fun_xk1, grad_fun_xk1);
		if (norm_grad < epsilon2)
			break;

		if(k == 0) {
			d.x = -grad_fun_xk1.x;
			d.y = -grad_fun_xk1.y;
		} else {
			const point2d_t grad_fun_xk = grad_fun(xk);
			const double beta = dotProduct2d(grad_fun_xk1, grad_fun_xk1) / dotProduct2d(grad_fun_xk, grad_fun_xk);
			d.x = beta * d.x - grad_fun_xk1.x;
			d.y = beta * d.y - grad_fun_xk1.y;
		}

		xk = xk1;
		const double t = findT_phi(xk, d);
		const point2d_t xk1_minus_xk = { t * d.x, t * d.y };
		xk1.x = xk.x + xk1_minus_xk.x;
		xk1.y = xk.y + xk1_minus_xk.y;

		const double norm_xk1_minus_xk = dotProduct2d(xk1_minus_xk, xk1_minus_xk);
		const double abs_fk1_minus_fk  = fabs(fun(xk1) - fun(xk));
		if (norm_xk1_minus_xk < epsilon2 && abs_fk1_minus_fk < epsilon) {
			if (is_seq) {
				break;
			} else {
				is_seq = true;
			}
		} else {
			is_seq = false;
		}
	}

	printf("Nonlinear Conjugate Gradient Method\nIters = %" PRIu32 "\n", k);

	return xk1;
}