// dp[i][j] = (s[i] == s[j]) and (j - i < 2 or dp[i + 1][j - 1])
//
// One thread per starting index i on a fixed diagonal (substring
// length). Cells on the same diagonal are mutually independent -
// each only reads dp[i+1][j-1] from the previous diagonal, which
// the host has already finished computing before launching this one.
extern "C" __global__
void palindrome_diagonal(const char* s, unsigned char* dp, int n, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = i + length - 1;
    if (i < 0 || j >= n) return;

    if (length == 1) {
        dp[i * n + j] = 1;
        return;
    }

    bool ends_match = s[i] == s[j];
    bool inner = (length == 2) || dp[(i + 1) * n + (j - 1)];
    dp[i * n + j] = (ends_match && inner) ? 1 : 0;
}
