__global__ void orcu_kernel18170(const int nrows, const int ndiags, int sbdiag, int ndofs, int* offsets, double* A, double* x, double* y) {
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int gsize=gridDim.x*blockDim.x;
  double ysum;
  int j, k, col, row;
  for (int i=tid; i<=nrows-1; i+=gsize) {
    {
      ysum=0.0;
      for (j=0; j<=ndiags-1; j++ ) {
        row=i+j*sbdiag;
        col=(floor((float)i/ndofs)+offsets[j])*ndofs;
        if (col>=0&&col<nrows) 
          for (k=0; k<=ndofs-1; k++ ) 
            ysum=ysum+A[row+k*nrows]*x[col+k];
      }
      y[i]=ysum;
    }
  }
}
void MatMult_SeqDIA(double* A, double* x, double* y, int M, int N, int P, int NOS, int DOF) {

  register int i,j,k;
  int col,row;
  double ysum;
  /*@ begin PerfTuning (
        def performance_params {
          param TC[]  = range(32,1025,32);
          param BC[]  = range(14,105,14);
          param PL[]  = [16,32,48];
        }
        def input_params {
          param M[] = [32];
          param N[] = [32];
          param P[] = [32];
          param NOS = 7;
          param DOF[] = range(1,17);
          constraint c1 = (M==N);
          constraint c2 = (N==P);
        }
        def input_vars {
          decl dynamic double A[M*N*P*DOF*DOF*NOS] = random;
          decl dynamic double x[M*N*P*DOF]         = random;
          decl dynamic double y[M*N*P*DOF]         = 0;
          decl static  int offsets[NOS]            = {-M*N*DOF,-M*DOF,-DOF,0,DOF,M*DOF,M*N*DOF};
        }
  ) @*/

/**-- (Generated by Orio) 
Best performance cost: 
  [0.94563200000000003, 0.92796800000000002, 0.92150399999999999, 0.91615999999999997, 0.92249599999999998] 
Tuned for specific problem sizes: 
  DOF = 9 
  M = 32 
  N = 32 
  NOS = 7 
  P = 32 
Best performance parameters: 
  BC = 98 
  PL = 32 
  TC = 960 
--**/



  int nrows=M*N*P*DOF;
  int ndiags=NOS;
  int ndofs=DOF;
  int sbdiag=M*N*P*DOF*DOF;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, preferL1Size=PL)

  for(i=0; i<=nrows-1; i++){
    ysum = 0.0;
    for(j=0; j<=ndiags-1; j++){
      row = i+j*sbdiag;
      col = (floor((float)i/ndofs)+offsets[j])*ndofs;
      if(col>=0&&col<nrows)
        for(k=0; k<=ndofs-1; k++)
          ysum += A[row+k*nrows] * x[col+k];
    }
    y[i] = ysum;
  }

  ) @*/
  {
    cudaDeviceSynchronize();
    /*declare variables*/
    double *dev_A, *dev_x, *dev_y;
    int *dev_offsets;
    int nthreads=960;
    /*calculate device dimensions*/
    dim3 dimGrid, dimBlock;
    dimBlock.x=nthreads;
    dimGrid.x=98;
    /*allocate device memory*/
    cudaMalloc(&dev_A,M *N *P *DOF *DOF *NOS*sizeof(double));
    cudaMalloc(&dev_x,M *N *P *DOF*sizeof(double));
    cudaMalloc(&dev_y,M *N *P *DOF*sizeof(double));
    cudaMalloc(&dev_offsets,NOS*sizeof(int));
    cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
    /*copy data from host to device*/
    cudaEventRecord(tstart,0);
    cudaMemcpy(dev_A,A,M *N *P *DOF *DOF *NOS*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x,x,M *N *P *DOF*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_offsets,offsets,NOS*sizeof(int),cudaMemcpyHostToDevice);
    cudaEventRecord(tstop,0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&orcu_transfer,tstart,tstop);
    cudaEventRecord(start,0);
    /*invoke device kernel*/
    orcu_kernel18170<<<dimGrid,dimBlock>>>(nrows,ndiags,sbdiag,ndofs,dev_offsets,dev_A,dev_x,dev_y);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&orcu_elapsed,start,stop);
    /*copy data from device to host*/
    cudaMemcpy(y,dev_y,M *N *P *DOF*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    /*free allocated memory*/
    cudaFree(dev_A);
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_offsets);
    cudaError_t err=cudaGetLastError();
    if (cudaSuccess!=err) 
      printf("CUDA runtime error: %s@",cudaGetErrorString(err));
  }
/*@ end @*/
  /*@ end @*/
}

