2. BLOCK_SIZE need to be multiple of 32 (wrap size)
3. - a. yes, a[i] accessing continuous memory locations on global memory
   - b. not applicable, a_s is shared memory
   - c. yes, b[j*blockDim.x*gridDim.x + i] is accessing continuous memory locations on global memory
   - d. no, c[i*4 + j] is accessing multiple of 4 memoty address, not conitnuous
   - e. not applicable, bc_s is shared memory
   - f. not applicable, a_s is shared memory
   - g. yes, d[i + 8] is accessing continuous memory locations on global memory
   - h. not applicable, bc_s is shared memory
   - i. no, e[i*8] is accessing multiple of 8 memory address, not continuous
4. - a. 0.125 OP/B
   - b. 4 OP/B
   - c. 4 OP/B, coarsening works inside shared memory, so it does not affect global memory access and FLOP