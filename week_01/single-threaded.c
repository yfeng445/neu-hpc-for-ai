#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *A;   
    int *B;   
    int m, k, n;
} Input;

typedef struct {
    int *C;   
    int m, n;
} Output;


Output matmul(Input t) {
    int *C = (int *)malloc((size_t)t.m * (size_t)t.n * sizeof(int));
    if (!C) { perror("malloc"); exit(1); }

    for (int i = 0; i < t.m; i++) {
        for (int j = 0; j < t.n; j++) {
            long sum = 0; 
            for (int p = 0; p < t.k; p++) {
                sum += (long)t.A[(size_t)i * t.k + p] * (long)t.B[(size_t)p * t.n + j];
            }
            C[(size_t)i * t.n + j] = (int)sum;
        }
    }
    Output o = { .C = C, .m = t.m, .n = t.n };
    return o;
}

void printMatrix(const int *M, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", M[(size_t)i * cols + j]);
        }
        printf("\n");
    }
}

/* ----- Tests ----- */
void loadTest(void) {
    {
        int A[1*1] = {2};
        int B[1*1] = {3};
        Input in = { .A = A, .B = B, .m = 1, .k = 1, .n = 1 };
        Output o = matmul(in);
        printf("1: 1x1 * 1x1\n");
        printMatrix(o.C, o.m, o.n);
        free(o.C);
        printf("\n");
    }

    {
        int A[1*1] = {4};
        int B[1*5] = {1,2,3,4,5};
        Input in = { .A = A, .B = B, .m = 1, .k = 1, .n = 5 };
        Output o = matmul(in);
        printf("2: 1x1 * 1x5\n");
        printMatrix(o.C, o.m, o.n);
        free(o.C);
        printf("\n");
    }

    {
        int A[2*1] = {1, 2};
        int B[1*3] = {3, 4, 5};
        Input in = { .A = A, .B = B, .m = 2, .k = 1, .n = 3 };
        Output o = matmul(in);
        printf("3: 2x1 * 1x3\n");
        printMatrix(o.C, o.m, o.n);
        free(o.C);
        printf("\n");
    }

    {
        int A[2*2] = {1, 2,
                      3, 4};
        int B[2*2] = {5, 6,
                      7, 8};
        Input in = { .A = A, .B = B, .m = 2, .k = 2, .n = 2 };
        Output o = matmul(in);
        printf("4: 2x2 * 2x2\n");
        printMatrix(o.C, o.m, o.n);
        free(o.C);
        printf("\n");
    }

    {
        int A[3*2] = {1,2,
                      3,4,
                      5,6};
        int B[2*4] = {7,8,9,10,
                      11,12,13,14};
        Input in = { .A = A, .B = B, .m = 3, .k = 2, .n = 4 };
        Output o = matmul(in);
        printf("5: 3x2 * 2x4\n");
        printMatrix(o.C, o.m, o.n);
        free(o.C);
        printf("\n");
    }
}

int main(void) {
    loadTest();
    return 0;
}
