#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdbool.h>
#include <math.h>

typedef struct point2d {
	double x, y;
} point2d_t;

typedef struct mat2x2 {
	double a11, a12, a21, a22;
} mat2x2_t;

double dotProduct2d(point2d_t x, point2d_t y);
point2d_t mat_vec2d(double alpha, mat2x2_t mat, point2d_t x);
point2d_t scalarmul_vec2d(double alpha, point2d_t x);
point2d_t add_vec2d(point2d_t x, point2d_t y);
double det_mat2x2(mat2x2_t mat);
mat2x2_t inversed_mat2x2(mat2x2_t mat);
bool isPositiveDefMat2x2(mat2x2_t mat);
double fun(point2d_t x);
point2d_t grad_fun(point2d_t x);
mat2x2_t fun_Hessian_mat2x2();
double findT_phi(point2d_t x, point2d_t grad_fun_x);
point2d_t nonlinearConjugateGradientMethod(point2d_t x0, double epsilon, uint32_t maxIter);
point2d_t methodNewtonRaphson(point2d_t x0, double epsilon, uint32_t maxIter);

int32_t main() {
	const point2d_t x0     = { -4.0, -2.0 };
	const double epsilon   = 1.0e-2;
	const uint32_t maxIter = 100;
	const uint8_t test     = 1; // 1 - Nonlinear Conjugate Gradient Method
	                            // 2 - method Newton-Raphson

	point2d_t xmin;
	switch (test) {
	case 1 :
		printf("Nonlinear Conjugate Gradient Method\n");
		xmin = nonlinearConjugateGradientMethod(x0, epsilon, maxIter);
		break;
	case 2 :
		printf("Newton-Raphson Method\n");
		xmin = methodNewtonRaphson(x0, epsilon, maxIter);
		break;
	default:
		printf("Error! value test\n");
		system("pause");
		return -1;
	}

	printf("\nXmin = (%.2f, %.2f)\n", xmin.x, xmin.y);

	system("pause");

	return 0;
}

inline double dotProduct2d(const point2d_t x, const point2d_t y) {
	return x.x * y.x + x.y * y.y;
}

inline point2d_t mat_vec2d(const double alpha, const mat2x2_t mat, const point2d_t x) {
	const point2d_t vec = { alpha * (mat.a11 * x.x + mat.a12 * x.y),
	                        alpha * (mat.a21 * x.x + mat.a22 * x.y) };
	return vec;
}

inline point2d_t scalarmul_vec2d(const double alpha, const point2d_t x) {
	const point2d_t vec = { alpha * x.x, alpha * x.y };
	return vec;
}

inline point2d_t add_vec2d(const point2d_t x, const point2d_t y) {
	const point2d_t vec = { x.x + y.x, x.y + y.y };
	return vec;
}

inline double det_mat2x2(const mat2x2_t mat) {
	return mat.a11 * mat.a22 - mat.a21 * mat.a12;
}

inline mat2x2_t inversed_mat2x2(const mat2x2_t mat) {
	const double det_mat = det_mat2x2(mat);
	if (det_mat == 0.0) {
		printf("Determinant = 0!\n");
		const mat2x2_t inv_mat2x2 = { 0, 0, 0, 0 };
		return inv_mat2x2;
	}
	const double inv_det_mat  = 1.0 / det_mat;
	const mat2x2_t inv_mat2x2 = { mat.a22 * inv_det_mat, -mat.a12 * inv_det_mat,
		                         -mat.a21 * inv_det_mat,  mat.a11 * inv_det_mat };
	return inv_mat2x2;
}

inline bool isPositiveDefMat2x2(const mat2x2_t mat) {
	// Sylvester's criterion
	if (mat.a11 <= 0.0 || det_mat2x2(mat) <= 0.0) {
		printf("No Positive-definite matrix!\n");
		return false;
	} else {
		return true;
	}
}

inline double fun(const point2d_t x) {
	return (x.x - 2.0 * x.y) * (x.x - 2.0 * x.y) + (x.x + 5.0) * (x.x + 5.0);
}

inline point2d_t grad_fun(const point2d_t x) {
	const point2d_t grad = { 4.0 * x.x - 4.0 * x.y + 10.0,
		                     8.0 * x.y - 4.0 * x.x         };
	return grad;
}

inline mat2x2_t fun_Hessian_mat2x2() {
	const mat2x2_t H = { 4.0, -4.0, -4.0, 8.0 };
	return H;
}

inline double findT_phi(const point2d_t x, const point2d_t d) {
	// Quadratic function
	//if (dotProduct2d(d, d) < 0) {
	//	printf("Error! d^phi/dt^2 < 0\n");
	//	return 0;
	//}
	const double A = (d.x - 2.0 * d.y);
	return - (A * (x.x - 2.0 * x.y) + (5.0 + x.x) * d.x) / (A * A + d.x * d.x);
}

point2d_t nonlinearConjugateGradientMethod(const point2d_t x0, const double epsilon, const uint32_t maxIter) {
	const double epsilon2 = epsilon * epsilon;
	point2d_t xk1 = x0;
	point2d_t xk, d;
	double dot_grad_xk, convergence;
	bool is_seq = false;
	uint32_t k = 0;

	for (k; k < maxIter; ++k) {
		const point2d_t grad_fun_xk1 = grad_fun(xk1);
		const double    dot_grad_xk1 = dotProduct2d(grad_fun_xk1, grad_fun_xk1);
		if (dot_grad_xk1 < epsilon2) {
			printf("grad f(xk) = %.e < epsilon = %.2e\n", sqrt(dot_grad_xk1), epsilon);
			break;
		}

		if (k == 0) {
			d = scalarmul_vec2d(-1.0, grad_fun_xk1);
		} else {
			const double beta = dot_grad_xk1 / dot_grad_xk;
			d = add_vec2d(scalarmul_vec2d(beta, d), scalarmul_vec2d(-1.0, grad_fun_xk1));
		}

		xk = xk1;
		const double t = findT_phi(xk, d);
		const point2d_t xk1_minus_xk = scalarmul_vec2d(t, d);
		xk1 = add_vec2d(xk1, xk1_minus_xk);

		convergence = dotProduct2d(xk1_minus_xk, xk1_minus_xk);
		const double abs_fk1_minus_fk = fabs(fun(xk1) - fun(xk));
		if (convergence < epsilon2 && abs_fk1_minus_fk < epsilon) {
			if (is_seq) {
				break;
			} else {
				is_seq = true;
			}
		} else {
			is_seq = false;
		}
		dot_grad_xk = dot_grad_xk1;
	}

	printf("Iters = %" PRIu32 "\nConvergence = %.2e\n", k, sqrt(convergence));

	return xk1;
}

point2d_t methodNewtonRaphson(const point2d_t x0, const double epsilon, const uint32_t maxIter) {
	const double epsilon2 = epsilon * epsilon;
	point2d_t xk1 = x0;
	bool is_seq = false;
	uint32_t k = 0;
	double convergence;

	for (k; k < maxIter; ++k) {
		const point2d_t grad_fun_xk1 = grad_fun(xk1);
		const double    dot_grad_xk1 = dotProduct2d(grad_fun_xk1, grad_fun_xk1);
		if (dot_grad_xk1 < epsilon2) {
			printf("grad f(xk) = %.e < epsilon = %.2e\n", sqrt(dot_grad_xk1), epsilon);
			break;
		}

		const mat2x2_t H    = fun_Hessian_mat2x2();
		const mat2x2_t invH = inversed_mat2x2(H);
		const point2d_t d   = isPositiveDefMat2x2(invH) ? mat_vec2d(-1.0, invH, grad_fun_xk1)
		                                                : scalarmul_vec2d(-1.0, grad_fun_xk1);

		const point2d_t xk = xk1;
		const double t = findT_phi(xk, d);
		const point2d_t xk1_minus_xk = scalarmul_vec2d(t, d);
		xk1 = add_vec2d(xk1, xk1_minus_xk);

		convergence = dotProduct2d(xk1_minus_xk, xk1_minus_xk);
		const double abs_fk1_minus_fk = fabs(fun(xk1) - fun(xk));
		if (convergence < epsilon2 && abs_fk1_minus_fk < epsilon) {
			if (is_seq) {
				break;
			}
			else {
				is_seq = true;
			}
		}
		else {
			is_seq = false;
		}
	}

	printf("Iters = %" PRIu32 "\nConvergence = %.2e\n", k, sqrt(convergence));

	return xk1;
}