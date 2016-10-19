#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

typedef struct point2d {
	double x;
	double y;
} point2d_t;

double fun(point2d_t x);
point2d_t grad_fun(point2d_t x);
point2d_t nonlinearConjugateGradientMethod(point2d_t x, double epsilon, uint32_t maxIter);

int32_t main() {

	system("pause");

	return 0;
}

inline double fun(const point2d_t x) {
	return (x.x - 2.0 * x.y) * (x.x - 2.0 * x.y) + (x.x + 5.0) * (x.x + 5.0);
}

inline point2d_t grad_fun(const point2d_t x) {
	point2d_t grad = { 4.0 * x.x + 4.0 * x.y + 10.0,
					   4.0 * x.x - 8.0 * x.y         };
	return grad;
}

point2d_t nonlinearConjugateGradientMethod(const point2d_t x0, const double epsilon,
                                           const uint32_t maxIter) {
	point2d_t grad_fun_x = grad_fun(x0);
	const double epsilon2 = epsilon * epsilon;
	double norm_grad = grad_fun_x.x * grad_fun_x.x + grad_fun_x.y * grad_fun_x.y;
	if (norm_grad < epsilon2)
		return x0;

	point2d_t d = { -grad_fun_x.x, -grad_fun_x.y };

	for (uint32_t k = 0; k < maxIter; ++k) {

	}

	return ;
}