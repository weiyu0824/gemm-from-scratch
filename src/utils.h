#ifndef GEMM_UTILS_H
#define GEMM_UTILS_H

#define CEIL_DIV(numerator, denominator) (((numerator) + (denominator) - 1) / (denominator))
#define INDEX_2D(row, col, width) (row * (width) + col) // row-major matrix

#endif