#include "exchange.h"

// Exchange interaction.
// M is unnormalized (in Tesla).

extern "C" __global__ void
exchange(float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
         float* __restrict__ Mx, float* __restrict__ My, float* __restrict__ Mz,
         float wx, float wy, float wz,
         int N0, int N1, int N2){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j >= N1 || k >= N2){
		return;
	}

	for(int i=0; i<N0; i++){

		float3 m, m1, m2, H;
		int I = idx(i, j, k);
		loadm(m, I);
		float Bsat = len(m);
		if (Bsat == 0) { Bsat = 1; }
		
		loadm(m1, idx(i, clamp(j+1,N1), k));
		loadm(m2, idx(i, clamp(j-1,N1), k));
		H = (wy/Bsat) * ((m1-m) + (m2-m));

		loadm(m1, idx(i, j, clamp(k+1,N2)));
		loadm(m2, idx(i, j, clamp(k-1,N2)));
		H += (wz/Bsat) * ((m1-m) + (m2-m));

		loadm(m1, idx(clamp(i+1,N0), j, k));
		loadm(m2, idx(clamp(i-1,N0), j, k));
		H  += (wx/Bsat) * ((m1-m) + (m2-m));

		Hx[I] = H.x;
		Hy[I] = H.y;
		Hz[I] = H.z;
	}
}

