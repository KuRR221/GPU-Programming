#include <cuda.h>
#include <stdio.h>
#include <windows.h>

// Defining useful values
#define PI 3.14159265358979323846f
#define BIN_WIDTH 0.25f
#define MAX_ANGLE 180.0f
#define NUM_BINS (int)(MAX_ANGLE / BIN_WIDTH)

int getDevice(int deviceno);

// Kernel to calculate angular separations and update histograms
__global__ void computeAngles(float* alpha1, float* delta1, float* alpha2, float* delta2, int num1, int num2, int* histogram) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx1 < num1 && idx2 < num2) {
        // Assigning variables
        float a1 = alpha1[idx1], d1 = delta1[idx1];
        float a2 = alpha2[idx2], d2 = delta2[idx2];

        // Calculate angular separation using the formula
        float cos_theta = sin(d1) * sin(d2) + cos(d1) * cos(d2) * cos(a1 - a2);
        float theta = acosf(fmaxf(-1.0f, fminf(cos_theta, 1.0f))) * (MAX_ANGLE / PI);

        // Determine the bin index
        int bin_idx = (int)(theta / BIN_WIDTH);
        if (bin_idx < NUM_BINS) {
            atomicAdd(&histogram[bin_idx], 1); // Safely update the histogram
        }
    }
}

// Function to convert arcminutes to radians
float arcminutesToRadians(float value) {
    return value * PI / (180.0f * 60.0f);
}

int main(int argc, char** argv) {
    SYSTEMTIME time;
    double kerneltime;

    // Check for correct input
    if (argc < 3) {
        printf("Usage: %s <real_data_file> <simulated_data_file>\n", argv[0]);
        return -1;
    }

    // Start premade getDevice function to print GPU model and specifications
    if (getDevice(0) != 0 ) {
        return -1;
    }

    // Read input data files
    FILE* realFile = fopen(argv[1], "r");
    FILE* simulatedFile = fopen(argv[2], "r");
    if (!realFile || !simulatedFile) {
        printf("Error: Unable to open input files.\n");
        return -1;
    }

    int NoofReal, NoofSim;
    fscanf(realFile, "%d", &NoofReal);
    fscanf(simulatedFile, "%d", &NoofSim);

    // Allocating memory for arrays
    float* ra_real = (float*)malloc(NoofReal * sizeof(float));
    float* decl_real = (float*)malloc(NoofReal * sizeof(float));
    float* ra_sim = (float*)malloc(NoofSim * sizeof(float));
    float* decl_sim = (float*)malloc(NoofSim * sizeof(float));

    // Updating arrays and converting values to radians
    for (int i = 0; i < NoofReal; ++i) {
        fscanf(realFile, "%f %f", &ra_real[i], &decl_real[i]);
        ra_real[i] = arcminutesToRadians(ra_real[i]);
        decl_real[i] = arcminutesToRadians(decl_real[i]);
    }

    for (int i = 0; i < NoofSim; ++i) {
        fscanf(simulatedFile, "%f %f", &ra_sim[i], &decl_sim[i]);
        ra_sim[i] = arcminutesToRadians(ra_sim[i]);
        decl_sim[i] = arcminutesToRadians(decl_sim[i]);
    }

    fclose(realFile);
    fclose(simulatedFile);

    // Allocate memory on GPU
    float *d_ra_real, *d_decl_real, *d_ra_sim, *d_decl_sim;
    int *d_histogram_DD, *d_histogram_DR, *d_histogram_RR;
    int histogram_DD[NUM_BINS] = {0};
    int histogram_DR[NUM_BINS] = {0};
    int histogram_RR[NUM_BINS] = {0};

    cudaMalloc((void**)&d_ra_real, NoofReal * sizeof(float));
    cudaMalloc((void**)&d_decl_real, NoofReal * sizeof(float));
    cudaMalloc((void**)&d_ra_sim, NoofSim * sizeof(float));
    cudaMalloc((void**)&d_decl_sim, NoofSim * sizeof(float));
    cudaMalloc((void**)&d_histogram_DD, NUM_BINS * sizeof(int));
    cudaMalloc((void**)&d_histogram_DR, NUM_BINS * sizeof(int));
    cudaMalloc((void**)&d_histogram_RR, NUM_BINS * sizeof(int));

    cudaMemcpy(d_ra_real, ra_real, NoofReal * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_decl_real, decl_real, NoofReal * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ra_sim, ra_sim, NoofSim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_decl_sim, decl_sim, NoofSim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram_DD, 0, NUM_BINS * sizeof(int));
    cudaMemset(d_histogram_DR, 0, NUM_BINS * sizeof(int));
    cudaMemset(d_histogram_RR, 0, NUM_BINS * sizeof(int));

    // Define thread and block dimensions
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks_DD((NoofReal + threadsPerBlock.x - 1) / threadsPerBlock.x, (NoofReal + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dim3 numBlocks_DR((NoofReal + threadsPerBlock.x - 1) / threadsPerBlock.x, (NoofSim + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dim3 numBlocks_RR((NoofSim + threadsPerBlock.x - 1) / threadsPerBlock.x, (NoofSim + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //Starting timer to measure time in kernel
    kerneltime = 0.0;
    GetSystemTime(&time);
    double start = time.wSecond + time.wMilliseconds / 1000.0;

    // Launch kernel for histogram_DD
    computeAngles<<<numBlocks_DD, threadsPerBlock>>>(d_ra_real, d_decl_real, d_ra_real, d_decl_real,
                                                                NoofReal, NoofReal, d_histogram_DD);
    cudaMemcpy(histogram_DD, d_histogram_DD, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Launch kernel for histogram_DR
    computeAngles<<<numBlocks_DR, threadsPerBlock>>>(d_ra_real, d_decl_real, d_ra_sim, d_decl_sim,
                                                                NoofReal, NoofSim, d_histogram_DR);
    cudaMemcpy(histogram_DR, d_histogram_DR, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Launch kernel for histogram_RR
    computeAngles<<<numBlocks_RR, threadsPerBlock>>>(d_ra_sim, d_decl_sim, d_ra_sim, d_decl_sim,
                                                                NoofSim, NoofSim, d_histogram_RR);
    cudaMemcpy(histogram_RR, d_histogram_RR, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    //Stopping timer after kernels have ran and printing the runtime
    GetSystemTime(&time);
    double end = time.wSecond + time.wMilliseconds / 1000.0;
    kerneltime = end-start;
    printf("   Run time = %.3f secs\n",kerneltime);

    // Write results to files, checking histogram sums and calculating omegas
    FILE* ddFile = fopen("dd_hist.txt", "w");
    FILE* drFile = fopen("dr_hist.txt", "w");
    FILE* rrFile = fopen("rr_hist.txt", "w");
    FILE* omegaFile = fopen("calc_omegas.txt", "w");
    long long sum_DD = 0, sum_DR = 0, sum_RR = 0;
    float omegas;

    //Iterating through histograms
    for (int i = 0; i < NUM_BINS; ++i) {
        
        if (histogram_RR[i] == 0) continue;
        omegas = (histogram_DD[i]- 2.0f*histogram_DR[i] + histogram_RR[i]) / histogram_RR[i];
        sum_DD += histogram_DD[i];
        sum_DR += histogram_DR[i];
        sum_RR += histogram_RR[i];

        fprintf(ddFile, "%.2f - %.2f: %d\n", i * BIN_WIDTH, (i + 1) * BIN_WIDTH, histogram_DD[i]);
        fprintf(drFile, "%.2f - %.2f: %d\n", i * BIN_WIDTH, (i + 1) * BIN_WIDTH, histogram_DR[i]);
        fprintf(rrFile, "%.2f - %.2f: %d\n", i * BIN_WIDTH, (i + 1) * BIN_WIDTH, histogram_RR[i]);
        fprintf(omegaFile, "%.2f - %.2f: %lf\n", i * BIN_WIDTH, (i + 1) * BIN_WIDTH, omegas);
    
    }

    // Adding histogram sums to the end of histogram files
    fprintf(ddFile, "Sum of DD histogram: %lld\n", sum_DD);
    fprintf(drFile, "Sum of DR histogram: %lld\n", sum_DR);
    fprintf(rrFile, "Sum of RR histogram: %lld\n", sum_RR);

    fclose(ddFile);
    fclose(drFile);
    fclose(rrFile);
    fclose(omegaFile);

    // Print histogram sums to console
    printf("Sum of DD histogram: %lld\n", sum_DD);
    printf("Sum of DR histogram: %lld\n", sum_DR);
    printf("Sum of RR histogram: %lld\n", sum_RR);

    // Free GPU memory
    cudaFree(d_ra_real);
    cudaFree(d_decl_real);
    cudaFree(d_ra_sim);
    cudaFree(d_decl_sim);
    cudaFree(d_histogram_DD);
    cudaFree(d_histogram_DR);
    cudaFree(d_histogram_RR);

    // Free host memory
    free(ra_real);
    free(decl_real);
    free(ra_sim);
    free(decl_sim);

    return 0;
}

int getDevice(int deviceNo) {
    int deviceCount;
    int device;
    cudaGetDeviceCount(&deviceCount);
    printf("   Found %d CUDA devices\n",deviceCount);
    
    if (deviceCount < 0 || deviceCount > 128) {
        return-1;
    } 
    
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("      Device %s                  device %d\n", deviceProp.name,device);
        printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
        printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
        printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
        printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
        printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
        printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
        printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
        printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
        printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
        printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
        printf("         maxGridSize                   =   %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("         maxThreadsDim in thread block =   %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("         concurrentKernels             =   ");

        if(deviceProp.concurrentKernels==1) {
            printf("     yes\n");
        } else {
            printf("    no\n");
        }

        printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
        
        if(deviceProp.deviceOverlap == 1) {
            printf("            Concurrently copy memory/execute kernel\n");
        }
    }
    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);

    if (device != deviceNo) {
        printf("   Unable to set device %d, using device %d instead",deviceNo, device);
    } else {
        printf("   Using CUDA device %d\n\n", device);
    }
    
return 0;
}
