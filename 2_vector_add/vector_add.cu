#include <cuda_fp16.h>

// block
// c = a + b
__global__ void vector_add(int* a, int* b, int* c, size_t N,
                           size_t tb = 1) {
    size_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t d_idx = t_idx * tb;
    for (size_t i = d_idx; i < d_idx + tb; ++i) {
        if (i >= N) {
            break;
        }

        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_vec(const int* a, const int* b, int* c, size_t N) {
    size_t idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int4* da = reinterpret_cast<const int4*>(&(a[idx]));
    const int4* db = reinterpret_cast<const int4*>(&(b[idx]));
    int4* dc = reinterpret_cast<int4*>(&(c[idx]));

    if (idx >= N) return; // TODO: 有bug

    dc->x = da->x + db->x;
    dc->y = da->y + db->y;
    dc->z = da->z + db->z;
    dc->w = da->w + db->w;
}

void vector_add_gpu(int* a, int* b, int* c, size_t N) {
    // TODO: 
    const size_t block_size = 1024;
    const size_t tb = 4;
    const size_t grid_size = N / (block_size * tb) + 1;

    // vector_add<<<grid_size, block_size>>>(a, b, c, N, tb);
    vector_add_vec<<<grid_size, block_size>>>(a, b, c, N);
}

