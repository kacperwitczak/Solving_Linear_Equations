import matplotlib.pyplot as plt

PATH = "D:\\Solving_Linear_Equations\\results\\"

def read_file_to_array(filename):
    array = []
    with open(filename, "r") as file:
        line = file.readline()

    data = line.rstrip('\n')
    array = data.split(',')
    result = [float(val) for val in array]

    return result

def plot_jacobi_gauss(jacobi, gauss):
    y_jacobi = jacobi
    x_jacobi = [i+1 for i in range(len(jacobi))]

    y_gauss = gauss
    x_gauss = [i+1 for i in range(len(gauss))]

    plt.semilogy(x_jacobi, y_jacobi, label='Jacobi')
    plt.semilogy(x_gauss, y_gauss, label='Gauss-Seidel')

    plt.xlabel('Iteracja')
    plt.ylabel('Norma błedu rezydualnego')
    plt.title('Jacobi vs Gauss-Seidel')
    plt.legend()

    plt.grid(True)
    plt.show()

def plot_compare_methods(sizes, jacobi, gauss, lu):
    plt.plot(sizes, jacobi, label="Jacobi")
    plt.plot(sizes, gauss, label="Gauss-Seidel")
    plt.plot(sizes, lu, label="Rozkład LU")

    plt.xlabel('Rozmiar macierzy')
    plt.ylabel('Czas obliczeń [s]')
    plt.title('Jacobi vs Gauss-Seidel vs Rozkład LU')
    plt.legend()

    plt.grid(True)
    plt.show()

def main():
    jacobi_norms_task_b = read_file_to_array(PATH + "jacobi_task_b")
    gauss_norms_task_b = read_file_to_array(PATH + "gauss_seidel_task_b")
    jacobi_norms_task_c = read_file_to_array(PATH + "jacobi_task_c")
    gauss_norms_task_c = read_file_to_array(PATH + "gauss_seidel_task_c")
    benchmark_sizes = read_file_to_array(PATH + "benchmark_sizes")
    benchmark_jacobi = read_file_to_array(PATH + "benchmark_jacobi")
    benchmark_gauss = read_file_to_array(PATH + "benchmark_gauss")
    benchmark_lu = read_file_to_array(PATH + "benchmark_lu")

    print("Jacobi Norms (Task B):", jacobi_norms_task_b)
    print("Gauss-Seidel Norms (Task B):", gauss_norms_task_b)
    print("Jacobi Norms (Task C):", jacobi_norms_task_c)
    print("Gauss-Seidel Norms (Task C):", gauss_norms_task_c)
    print("Benchmark Sizes:", benchmark_sizes)
    print("Benchmark Jacobi:", benchmark_jacobi)
    print("Benchmark Gauss-Seidel:", benchmark_gauss)
    print("Benchmark LU Decomposition:", benchmark_lu)

    plot_jacobi_gauss(jacobi_norms_task_b, gauss_norms_task_b)
    plot_jacobi_gauss(jacobi_norms_task_c, gauss_norms_task_c)
    plot_compare_methods(benchmark_sizes, benchmark_jacobi, benchmark_gauss, benchmark_lu)


if __name__ == '__main__':
    main()