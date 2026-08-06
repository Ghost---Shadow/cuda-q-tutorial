// Brute-force pairwise check: one thread per (i, j) candidate pair,
// i < j, all mutually independent - unlike the CPU's two-pointer walk
// (a real sequential data dependency: which pointer to move next
// depends on the running sum so far), every pair here can be checked
// simultaneously with no shared state except the single result write.
// Trades the two-pointer's O(n) for O(n^2) work in exchange for zero
// dependencies between threads - the genuinely parallel substructure
// this problem offers a GPU, at the cost of not being the asymptotically
// optimal algorithm.
extern "C" __global__
void two_sum_pairs(const long long* numbers, int n, long long target, int* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n || i >= j) return;

    if (numbers[i] + numbers[j] == target) {
        result[0] = i;
        result[1] = j;
    }
}
