// brew install open-mpi to install on mac
//
// ./run_mpi_cpu.sh 00_mpi_cpu.c 1
// ./run_mpi_cpu.sh 00_mpi_cpu.c 2
// ./run_mpi_cpu.sh 00_mpi_cpu.c 3



#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    printf("Hello from rank %d of %d\n", rank, size);

    MPI_Finalize();
    return 0;
}
