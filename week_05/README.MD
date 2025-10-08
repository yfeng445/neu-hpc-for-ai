nvcc -O3 -arch=sm_80 -std=c++17 fa2_fwd.cu fa2_bwd.cu test.cu -o fa2_test.exe
./fa2_test.exe