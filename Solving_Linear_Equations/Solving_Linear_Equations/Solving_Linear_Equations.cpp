#include <iostream>
#include <vector>
#include <cassert>
#include <math.h>
#include <fstream>
#include <chrono>

const std::string path = "D:\\Solving_Linear_Equations\\results\\";

template<typename T>
void save_vector_to_file(const std::string& file_path, const std::vector<T>& vec) {
	std::ofstream file(file_path);

	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file");
	}

	for (size_t i = 0; i < vec.size(); ++i) {
		file << vec[i];
		if (i != vec.size() - 1) {
			file << ',';
		}
	}

	file << '\n';

	file.close();
}

class Matrix {
private:
	int rows;
	int cols;
	std::vector<std::vector<double>> matrix;

public:
	Matrix(int rows, int cols) : rows(rows), cols(cols), matrix(rows, std::vector<double>(cols, 0.0)) {

	}

	Matrix(int rows, int cols, double val) : rows(rows), cols(cols), matrix(rows, std::vector<double>(cols, val)) {

	}

	Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), matrix(other.matrix) {}

	//column vector
	Matrix(int N, int f, double(*func)(int, int)) : rows(N), cols(1), matrix(rows, std::vector<double>(cols, 0.0)) {
		for (int i = 0; i < rows; i++) {
			matrix[i][0] = func(i, f);
		}
	}

	Matrix(int N, double a1, double a2, double a3) : rows(N), cols(N), matrix(rows, std::vector<double>(cols, 0.0)) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				if (i == j) {
					matrix[i][j] = a1;
				}
				else if (abs(i - j) == 1) {
					matrix[i][j] = a2;
				}
				else if (abs(i - j) == 2) {
					matrix[i][j] = a3;
				}
				else {
					matrix[i][j] = 0;
				}
			}
		}
	}

	int get_rows() const {
		return rows;
	}

	int get_cols() const {
		return cols;
	}

	double get_value(int i, int j) const {
		assert(i < this->rows && j < this->cols && i > 0 && j > 0);

		return matrix[i][j];
	}

	void set_value(int i, int j, double value) {
		assert(i < this->rows && j < this->cols && i > 0 && j > 0);

		matrix[i][j] = value;
	}

	void print_matrix() const {
		for (const auto& row : matrix) {
			for (const auto& col : row) {
				std::cout << col << ' ';
			}
			std::cout << '\n';
		}
	}

	std::vector<double>& operator[](int index) {
		assert(index >= 0 && index < rows);
		return matrix[index];
	}

	const std::vector<double>& operator[](int index) const {
		assert(index >= 0 && index < rows);
		return matrix[index];
	}

	friend class MatrixArthmetic;
};

class MatrixArthmetic {
public:
	static Matrix add(const Matrix& a, const Matrix& b) {
		assert(a.get_rows() == b.get_rows() && a.get_cols() == b.get_cols());

		Matrix result(a.get_rows(), a.get_cols());
		for (int i = 0; i < a.get_rows(); i++) {
			for (int j = 0; j < a.get_cols(); j++) {
				result[i][j] = a[i][j] + b[i][j];
			}
		}

		return result;
	}

	static Matrix mul(const Matrix& a, const Matrix& b) {
		assert(a.get_cols() == b.get_rows());

		Matrix result(a.get_rows(), b.get_cols());
		for (int i = 0; i < a.get_rows(); i++) {
			for (int j = 0; j < b.get_cols(); j++) {
				double val = 0;
				for (int k = 0; k < a.get_cols(); k++) {
					val += a[i][k] * b[k][j];
				}
				result[i][j] = val;
			}
		}

		return result;
	}

	static Matrix mul_scalar(const Matrix& a, double v) {
		Matrix result(a.get_rows(), a.get_cols());

		for (int i = 0; i < a.get_rows(); i++) {
			for (int j = 0; j < a.get_cols(); j++) {
				result[i][j] = a[i][j] * v;
			}
		}

		return result;
	}

	static Matrix sub(const Matrix& a, const Matrix& b) {
		assert(a.get_rows() == b.get_rows() && a.get_cols() == b.get_cols());

		Matrix result(a.get_rows(), a.get_cols());
		for (int i = 0; i < a.get_rows(); i++) {
			for (int j = 0; j < a.get_cols(); j++) {
				result[i][j] = a[i][j] - b[i][j];
			}
		}

		return result;
	}
};

class Solver {
public:
	static Matrix residuum(const Matrix& A, const Matrix& x, const Matrix& b) {
		Matrix A_dot_x = MatrixArthmetic::mul(A, x);

		Matrix result = MatrixArthmetic::sub(A_dot_x, b);

		return result;
	}

	static double euclidean_norm(const Matrix& m) {
		assert(m.get_cols() == 1);
		double norm = 0;
		for (int i = 0; i < m.get_rows(); i++) {
			norm += pow(m[i][0], 2);
		}

		return sqrt(norm);
	}

	//source: Lecture "Wyklad 3"
	//The number of iterations is returned by reference!
	static std::vector<double> Gauss_Seidel_Method(const Matrix& A, const Matrix& x_init, const Matrix& B, int& max_iterations, double epsilon) {
		Matrix x(x_init);
		Matrix x_prev(x_init);

		std::vector<double> norms;

		int iterations = 0;

		while (iterations <= max_iterations) {
			iterations++;
			for (int i = 0; i < x.get_rows(); i++) {
				double sum1 = 0;
				for (int j = 0; j < i; j++) {
					sum1 += A[i][j] * x[j][0];
				}

				double sum2 = 0;
				for (int j = i + 1; j < x.get_rows(); j++) {
					sum2 += A[i][j] * x_prev[j][0];
				}

				x[i][0] = (B[i][0] - sum1 - sum2) / A[i][i];
			}
			x_prev = x;

			double norm = Solver::euclidean_norm(Solver::residuum(A, x, B));
			norms.push_back(norm);

			if (norm < epsilon) {
				break;
			}
		}

		max_iterations = iterations;

		return norms;
	}

	//source: Lecture "Wyklad 3"
	//The number of iterations is returned by reference!
	static std::vector<double> Jacobi_Method(const Matrix& A, const Matrix& x_init, const Matrix& B, int& max_iterations, double epsilon) {
		Matrix x(x_init);
		Matrix x_prev(x_init);

		std::vector<double> norms;

		int iterations = 0;

		while (iterations <= max_iterations) {
			iterations++;
			for (int i = 0; i < x.get_rows(); i++) {
				double sum1 = 0;
				for (int j = 0; j < i; j++) {
					sum1 += A[i][j] * x_prev[j][0];
				}

				double sum2 = 0;
				for (int j = i + 1; j < x.get_rows(); j++) {
					sum2 += A[i][j] * x_prev[j][0];
				}

				x[i][0] = (B[i][0] - sum1 - sum2) / A[i][i];
			}

			x_prev = x;

			double norm = Solver::euclidean_norm(Solver::residuum(A, x, B));
			norms.push_back(norm);

			if (norm < epsilon) {
				break;
			}
		}

		max_iterations = iterations;

		return norms;
	}

	static double LUdecomposition_Method(const Matrix& A, const Matrix& X, const Matrix& B) {
		int m = A.get_rows();
		Matrix U = Matrix(A);
		Matrix L = Matrix(m, 1, 0, 0);

		//source: Lecture "Wyklad 2"
		for (int i = 1; i < m; i++) {
			for (int j = 0; j < i; j++) {
				L[i][j] = U[i][j] / U[j][j];
				//from j because we want upper triangle
				for (int k = j + 1; k < m; k++) {
					U[i][k] -= L[i][j] * U[j][k];
				}

			}
		}


		// L x Y = B
		// U x X = B
		//source: https://eduinf.waw.pl/inf/alg/008_nm/0026.php
		Matrix Y = Matrix(m, 1, 1); //column vector
		Matrix x = Matrix(X);


		//forward-substitution
		for (int i = 0; i < m; i++) {
			double val = B[i][0];
			for (int k = 0; k < i; k++) {
				val -= L[i][k] * Y[k][0];
			}
			Y[i][0] = val;
		}


		//back-substitution
		for (int i = m - 1; i >= 0; i--) {
			double val = Y[i][0];
			for (int k = m - 1; k > i; k--) {
				val -= U[i][k] * x[k][0];
			}
			x[i][0] = val / U[i][i];
		}

		return Solver::euclidean_norm(Solver::residuum(A, x, B));
	}
};



struct Constants {
	int N;
	int c;
	int d;
	int e;
	int f;
};

Constants solve_index(long long index) {
	Constants result;
	result.d = index % 10;
	index /= 10;
	result.c = index % 10;
	index /= 10;
	result.e = index % 10;
	index /= 10;
	result.f = index % 10;

	result.N = 900 + result.c * 10 + result.d;

	return result;
}

double init_vector_func(int n, int f) {
	return sin(n * (f + 1));
}

void Task_B() {

	std::cout << "***************Task_B***************\n";
	Constants c = solve_index(193609);

	int N = c.N;
	int e = c.e;
	int f = c.f;

	double a1 = e + 5;
	double a2 = -1;
	double a3 = -1;

	const Matrix A(N, a1, a2, a3);
	const Matrix b(N, f, &init_vector_func);
	const Matrix x(N, 1, 1);

	int iterations_jacobi = 1000;
	auto start = std::chrono::high_resolution_clock::now();
	std::vector<double> jacobi_norms = Solver::Jacobi_Method(A, x, b, iterations_jacobi, 1e-9);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = end - start;
	double time_taken = duration.count();
	std::cout << "Matrix size: " << N << std::endl;
	std::cout << "Duration: " << time_taken << " seconds" << std::endl;
	std::cout << "Norm Jacobi: " << jacobi_norms.back() << std::endl;
	std::cout << "Iterations Jacobi: " << iterations_jacobi << std::endl;
	save_vector_to_file<double>(path + "jacobi_task_b", jacobi_norms);
	std::cout << '\n';

	int iterations_gauss_seidel = 1000;
	start = std::chrono::high_resolution_clock::now();
	std::vector<double> gauss_seidel_norms = Solver::Gauss_Seidel_Method(A, x, b, iterations_gauss_seidel, 1e-9);
	end = std::chrono::high_resolution_clock::now();

	duration = end - start;
	time_taken = duration.count();
	std::cout << "Matrix size: " << N << std::endl;
	std::cout << "Duration: " << time_taken << " seconds" << std::endl;
	std::cout << "Norm Gauss-Seidel: " << gauss_seidel_norms.back() << std::endl;
	std::cout << "Iterations Gauss-Seidel: " << iterations_gauss_seidel << std::endl;
	save_vector_to_file<double>(path + "gauss_seidel_task_b", gauss_seidel_norms);

	std::cout << "***********************************\n";
}

void Task_C() {
	std::cout << "***************Task_C***************\n";
	Constants c = solve_index(193609);

	int N = c.N;
	int e = c.e;
	int f = c.f;

	double a1 = 3;
	double a2 = -1;
	double a3 = -1;

	const Matrix A(N, a1, a2, a3);
	const Matrix b(N, f, &init_vector_func);
	const Matrix x(N, 1, 1);

	int iterations_jacobi = 1000;
	auto start = std::chrono::high_resolution_clock::now();
	std::vector<double> jacobi_norms = Solver::Jacobi_Method(A, x, b, iterations_jacobi, 1e-9);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = end - start;
	double time_taken = duration.count();
	std::cout << "Matrix size: " << N << std::endl;
	std::cout << "Duration: " << time_taken << " seconds" << std::endl;
	std::cout << "Norm jacobi: " << jacobi_norms.back() << std::endl;
	std::cout << "Iterations Jacobi: " << iterations_jacobi << std::endl;
	save_vector_to_file<double>(path + "jacobi_task_c", jacobi_norms);

	std::cout << '\n';

	int iterations_gauss_seidel = 1000;
	start = std::chrono::high_resolution_clock::now();
	std::vector<double> gauss_seidel_norms = Solver::Gauss_Seidel_Method(A, x, b, iterations_gauss_seidel, 1e-9);
	end = std::chrono::high_resolution_clock::now();

	duration = end - start;
	time_taken = duration.count();
	std::cout << "Matrix size: " << N << std::endl;
	std::cout << "Duration: " << time_taken << " seconds" << std::endl;
	std::cout << "Norm gauss_seidel: " << gauss_seidel_norms.back() << std::endl;
	std::cout << "Iterations Gauss-Seidel: " << iterations_gauss_seidel << std::endl;
	save_vector_to_file<double>(path + "gauss_seidel_task_c", gauss_seidel_norms);

	std::cout << "***********************************\n";
}

void Task_D() {
	std::cout << "***************Task_D***************\n";

	Constants c = solve_index(193609);

	int N = c.N;
	int e = c.e;
	int f = c.f;

	double a1 = 3;
	double a2 = -1;
	double a3 = -1;

	const Matrix A(N, a1, a2, a3);
	const Matrix b(N, f, &init_vector_func);
	const Matrix x(N, 1, 1);

	auto start = std::chrono::high_resolution_clock::now();
	double lu_norm = Solver::LUdecomposition_Method(A, x, b);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = end - start;
	double time_taken = duration.count();
	std::cout << "Matrix size: " << N << std::endl;
	std::cout << "Duration: " << time_taken << " seconds" << std::endl;
	std::cout << "Norm LU decomposition: " << lu_norm << std::endl;
	std::vector<double> norms_lu(lu_norm);

	std::cout << "***********************************\n";

}

void Task_E() {
	std::cout << "***************Task_E***************\n";
	std::vector<int> matrix_sizes = { 100,500,1000,1500,2000,2500,3000,3500, 4000, 4500, 5000, 5500, 6000 };

	Constants c = solve_index(193609);

	int e = c.e;
	int f = c.f;

	double a1 = e + 5;
	double a2 = -1;
	double a3 = -1;

	std::vector<double> jacobi_time;
	std::vector<double> gauss_time;
	std::vector<double> lu_time;

	for (const auto& size : matrix_sizes) {
		int N = size;
		const Matrix A(N, a1, a2, a3);
		const Matrix b(N, f, &init_vector_func);
		const Matrix x(N, 1, 1);

		{
			int iterations_jacobi = 1000;
			auto start_jacobi = std::chrono::high_resolution_clock::now();
			std::vector<double> jacobi_norms = Solver::Jacobi_Method(A, x, b, iterations_jacobi, 1e-9);
			auto end_jacobi = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration_jacobi = end_jacobi - start_jacobi;
			double time_taken_jacobi = duration_jacobi.count();
			jacobi_time.push_back(time_taken_jacobi);
			std::cout << "jacobi N: " << N << " time: " << time_taken_jacobi << '\n';
			std::cout << "Norm jacobi " << jacobi_norms.back() << std::endl << std::endl;
		}

		{
			int iterations_gauss_seidel = 1000;
			auto start_gauss = std::chrono::high_resolution_clock::now();
			std::vector<double> gauss_seidel_norms = Solver::Gauss_Seidel_Method(A, x, b, iterations_gauss_seidel, 1e-9);
			auto end_gauss = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration_gauss = end_gauss - start_gauss;
			double time_taken_gauss = duration_gauss.count();
			gauss_time.push_back(time_taken_gauss);
			std::cout << "gauss N: " << N << " time: " << time_taken_gauss << '\n';
			std::cout << "Norm gauss_seidel: " << gauss_seidel_norms.back() << std::endl << std::endl;

		}

		{
			auto start_lu = std::chrono::high_resolution_clock::now();
			double norm = Solver::LUdecomposition_Method(A, x, b);
			auto end_lu = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration_lu = end_lu - start_lu;
			double time_taken_lu = duration_lu.count();
			lu_time.push_back(time_taken_lu);
			std::cout << "LU N: " << N << " time: " << time_taken_lu << '\n';
			std::cout << "Norm LU decomposition: " << norm << std::endl << std::endl;
		}
	}

	save_vector_to_file(path + "benchmark_sizes", matrix_sizes);
	save_vector_to_file(path + "benchmark_jacobi", jacobi_time);
	save_vector_to_file(path + "benchmark_gauss", gauss_time);
	save_vector_to_file(path + "benchmark_lu", lu_time);


	std::cout << "***********************************\n";

}

int main() {
	//std::ios_base::sync_with_stdio(false);
	//std::cin.tie(NULL);
	Task_B();
	std::cout << '\n';
	Task_C();
	std::cout << '\n';
	Task_D();
	std::cout << '\n';
	Task_E();
	std::cout << '\n';


	return 0;
}