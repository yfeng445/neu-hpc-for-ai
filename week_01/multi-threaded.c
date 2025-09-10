
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct{
    int *A;  
    int *B; 
    int m, k, n;
} Input;

typedef struct{
    int *C;  
    int m, n;
} Output;

typedef struct {
    const Input *in;
    int *C;
    int row_start;  
    int row_end;    
} WorkerArgs;

static void* worker(void *arg) {
    WorkerArgs *w = (WorkerArgs*)arg;
    const Input *t = w->in;
    const int *A = t->A;
    const int *B = t->B;
    int *C = w->C;
    int m = t->m, k = t->k, n = t->n;

    for (int i = w->row_start; i < w->row_end; ++i) {
        const int *Ai = A + (size_t)i * k;
        int *Ci = C + (size_t)i * n;
        for (int j = 0; j < n; ++j) {
            long sum = 0;
            const int *Bj = B + j;  
            for (int p = 0; p < k; ++p) {
                sum += (long)Ai[p] * (long)Bj[(size_t)p * n];
            }
            Ci[j] = (int)sum;
        }
    }
    return NULL;
}


Output matmul_mt(Input t, int num_threads) {
    if (num_threads <= 0) num_threads = 1;
    if (num_threads > t.m) num_threads = t.m; 

    int *C = (int*)malloc((size_t)t.m * (size_t)t.n * sizeof(int));
    if (!C) { perror("malloc"); exit(1); }

    pthread_t *th = (pthread_t*)malloc((size_t)num_threads * sizeof(pthread_t));
    WorkerArgs *wa = (WorkerArgs*)malloc((size_t)num_threads * sizeof(WorkerArgs));
    if (!th || !wa) { perror("malloc"); exit(1); }

    int base = t.m / num_threads;
    int rem  = t.m % num_threads;
    int row  = 0;

    for (int i = 0; i < num_threads; ++i) {
        int take = base + (i < rem ? 1 : 0);
        wa[i].in = &t;
        wa[i].C = C;
        wa[i].row_start = row;
        wa[i].row_end   = row + take;
        row += take;
        pthread_create(&th[i], NULL, worker, &wa[i]);
    }
    for (int i = 0; i < num_threads; ++i) pthread_join(th[i], NULL);

    free(th);
    free(wa);

    Output o = {.C = C, .m = t.m, .n = t.n};
    return o;
}

void printMatrix(const int *M, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) printf("%d ", M[(size_t)i*cols + j]);
        printf("\n");
    }
}

void loadTest(int p) {
    {
        int A[1*1] = {2};
        int B[1*1] = {3};
        Input in = {.A=A,.B=B,.m=1,.k=1,.n=1};
        Output o = matmul_mt(in, p);
        printf("1: 1x1 * 1x1\n");
        printMatrix(o.C, o.m, o.n);
        free(o.C); puts("");
    }
    {
        int A[1*1] = {4};
        int B[1*5] = {1,2,3,4,5};
        Input in = {.A=A,.B=B,.m=1,.k=1,.n=5};
        Output o = matmul_mt(in, p);
        printf("2: 1x1 * 1x5\n");
        printMatrix(o.C, o.m, o.n);
        free(o.C); puts("");
    }
    {
        int A[2*1] = {1,2};
        int B[1*3] = {3,4,5};
        Input in = {.A=A,.B=B,.m=2,.k=1,.n=3};
        Output o = matmul_mt(in, p);
        printf("3: 2x1 * 1x3\n");
        printMatrix(o.C, o.m, o.n);
        free(o.C); puts("");
    }
    {
        int A[2*2] = {1,2, 3,4};
        int B[2*2] = {5,6, 7,8};
        Input in = {.A=A,.B=B,.m=2,.k=2,.n=2};
        Output o = matmul_mt(in, p);
        printf("4: 2x2 * 2x2\n");
        printMatrix(o.C, o.m, o.n);
        free(o.C); puts("");
    }
    {
        int A[3*2] = {1,2, 3,4, 5,6};
        int B[2*4] = {7,8,9,10, 11,12,13,14};
        Input in = {.A=A,.B=B,.m=3,.k=2,.n=4};
        Output o = matmul_mt(in, p);  
        printf("5: 3x2 * 2x4\n");
        printMatrix(o.C, o.m, o.n);
        free(o.C); puts("");
    }
}

int main(int argc, char **argv) {
    int device_cores = 8;
    int p = 1;

    if (argc >= 2) {
        int requested = atoi(argv[1]);
        if (requested >= 1 && requested <= device_cores) {
            p = requested;
        } else {
            fprintf(stderr, "Invalid thread count %s, Falling back to 1.\n", argv[1]);
            p = 1;
        }
    }

    printf("Running in %d thread(s).\n\n", p);
    loadTest(p);
    return 0;
}
