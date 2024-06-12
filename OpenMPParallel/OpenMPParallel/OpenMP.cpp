#include <iostream>
#include <omp.h>

#define N 512

using namespace std;


void mul_matrix(double** matrix_1, double** matrix_2, double** matrix_mul) {
#pragma omp parallel for shared(matrix_1, matrix_2, matrix_mul)
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			matrix_mul[i][j] = 0.0;
			for (int k = 0; k < N; k++)
				matrix_mul[i][j] += matrix_1[i][k] * matrix_2[k][j];
		}
}

double** get_matrix() {
	double** matrix = new double* [N];
	for (int i = 0; i < N; i++)
		matrix[i] = new double[N];

	return matrix;
}

double** fill_random_matrix(double** matrix) {
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			matrix[i][j] = rand() % 5;

	return matrix;
}

int main() {
	double** matrix_1, ** matrix_2, ** matrix_mul, temp_time;
	int before_count_thread, max_threads;

	cout << "Input before_count_thread = ";
	cin >> before_count_thread;

	max_threads = omp_get_num_procs();
	cout << "max_threads = " << max_threads << "\n";
	before_count_thread = min(max_threads, max(1, before_count_thread));
	matrix_1 = fill_random_matrix(get_matrix());
	matrix_2 = fill_random_matrix(get_matrix());

	for (int count_thread = 1; count_thread <= before_count_thread; count_thread++) {
		matrix_mul = get_matrix();
		omp_set_num_threads(count_thread);
		temp_time = omp_get_wtime();
		mul_matrix(matrix_1, matrix_2, matrix_mul);
		cout << "Count thread = " << count_thread << ", time = " << omp_get_wtime() - temp_time << "s\n";
	}
}