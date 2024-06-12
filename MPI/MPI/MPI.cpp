#include <iostream>
#include <mpi.h>

#define N 2000

using namespace std;


double* mul_matrix(double* matrix_1, double* matrix_2, int runk, int max_runk) {
	double* matrix_mul;
	matrix_mul = new double[N * N];

	for (int i = 0; i < N; i++)
		for (int j = runk; j < N; j += max_runk) {
			matrix_mul[i * N + j] = 0.0;
			for (int k = 0; k < N; k++)
				matrix_mul[i * N + j] += matrix_1[i * N + k] * matrix_2[k * N + j];
		}

	return matrix_mul;
}

int main(int argc, char* argv[]) {
	double* matrix_1, * matrix_2, * matrix_mul, * matrix_result, temp_time;
	int runk, max_runk;

	matrix_1 = new double[N * N];
	matrix_2 = new double[N * N];
	matrix_result = new double[N * N];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &runk);
	MPI_Comm_size(MPI_COMM_WORLD, &max_runk);

	if (runk == 0) {
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				matrix_1[i * N + j] = matrix_2[i * N + j] = 2;
		temp_time = MPI_Wtime();
		for (int temp_runk = 1; temp_runk < max_runk; temp_runk++) {
			MPI_Send(matrix_1, N * N, MPI_DOUBLE, temp_runk, 1, MPI_COMM_WORLD);
			MPI_Send(matrix_2, N * N, MPI_DOUBLE, temp_runk, 2, MPI_COMM_WORLD);
		}
	}
	else {
		MPI_Recv(matrix_1, N * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(matrix_2, N * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	matrix_mul = mul_matrix(matrix_1, matrix_2, runk, max_runk);
	MPI_Reduce(matrix_mul, matrix_result, N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (runk == 0)
		cout << "Count thread = " << max_runk << ", time = " << MPI_Wtime() - temp_time << "s\n";

	MPI_Finalize();
}