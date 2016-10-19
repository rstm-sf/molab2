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
double grad_fun(point2d_t x);

uint32_t main() {

	system("pause");

	return 0;
}

inline double fun(const point2d_t x) {
	return (x.x - 2.0 * x.y) * (x.x - 2.0 * x.y) + (x.x + 5.0) * (x.x + 5.0);
}

inline double grad_fun(const point2d_t x) {
	return 20 + 8 * x.y;
}