#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<time.h>

#pragma pack (1)
//设置对齐方式

#define GauSize 5

typedef struct{
    short type;         //文件类型，必须为BM
    int size;           //整个位图文件的大小，以字节为单位
    short reserved1;    //保留，全0
    short reserved2;    //保留，全0
    int offset;         //位图数据的起始位置，字节为单位
}BMPHeader;

typedef struct 
{
    int size;       //BMP 信息头的大小，字节为单位
    int width;      //宽度，以像素为单位
    int height;     //高度，以像素为单位
    short planes;   // 一般为1
    short bitsPerPixel; //每个像素所用的位数
    unsigned compression; //压缩类型 0：不压缩   1：BI_RLE8压缩类型  2：BI_RLE4压缩类型
    unsigned imageSize;   //位图数据部分的大小，以字节为单位
    int xPelsPerMeter;    //水平分辨率，每米像素数，可置为0
    int yPelsPerMeter;    //同上
    int clrUsed;          //位图实际使用的颜色表中的颜色数，置为0都使用
    int clrImportant;     //位图显示时重要的颜色数，置为0都重要
}BMPInfoHeader;

// typedef struct {
//     unsigned char x,y,z; //bmp中24位位图数据存储的单个像素通道顺序是bgr ， 且每行必须为4的倍数， 不够用0填充
// }uchar3;

void loadBMPFile(uchar3 ** dst, int *width , int *height , const char *name){

    BMPHeader hdr;
    BMPInfoHeader infoHdr;
    int x,y;
    FILE *fd ;
    

    printf("Loading %s ...\n", name);
    if(sizeof(uchar3) !=3){
        printf("***** uchaar3 is not 3 bytes *****\n");
        return ;
    }
    if( !(fd = fopen(name, "rb"))){
        printf("***** fail to open %s *****\n", name);
        return ;
    }


    fread(&hdr , sizeof(hdr) , 1, fd);

    if(hdr.type != 0x4d42){
        printf("***** it is not a bmp file *****\n");
        return ;
    }

    fread(&infoHdr , sizeof(infoHdr) , 1, fd);

    if(infoHdr.bitsPerPixel !=24){
        printf("***** invalid color depth (24 bits needed) *****\n");
        printf("It is %hd\n" , infoHdr.bitsPerPixel);
        printf("size of short : %d\n",sizeof(short));
        printf("size of hdr is %d , infoHdr is %d\n" , sizeof(hdr) , sizeof(infoHdr));
        return;
    }
    printf("size of short : %d\n",sizeof(short));
    printf("size of hdr is %d , infoHdr is %d\n" , sizeof(hdr) , sizeof(infoHdr));
    if(infoHdr.compression){
        printf("***** cannot solve copressed image *****\n");
        return ;
    }


    *width = infoHdr.width;
    *height = infoHdr.height;
    //malloc space for image data 
    *dst = (uchar3 *)malloc(*width * *height * sizeof(uchar3));
    // cudaMallocManaged(dst , *width * *height * sizeof(uchar3));
    printf("image width: %u\n", infoHdr.width);
    printf("image height: %u\n", infoHdr.height);

    fseek(fd, hdr.offset - sizeof(hdr) - sizeof(infoHdr) , SEEK_CUR);

    for(y = 0 ; y < infoHdr.height ; y++){
        for(x = 0 ; x < infoHdr.width ; x++){
            (*dst)[ y * infoHdr.width + x ].z = fgetc(fd);
            (*dst)[ y * infoHdr.width + x ].y = fgetc(fd);
            (*dst)[ y * infoHdr.width + x ].x = fgetc(fd);

        }

        //pass filling bytes
        for(x = 0 ; x < (4 - (3*infoHdr.width)%4)%4 ; x++ ){
            fgetc(fd);
        }
    }

    printf("image file loaded successful! \n");

    fclose(fd);


}

void saveBMPFile(uchar3 *dst , int width , int height , const char *name){
    printf("in save bmp file\n");
    BMPHeader hdr; 
    BMPInfoHeader infoHdr;
     
    hdr.type = 0x4d42;
    hdr.reserved1 = 0;
    hdr.reserved2 = 0; 
    hdr.offset = 54;
    hdr.size = hdr.offset + 3 * height * (width + (4-(width*3)%4)%4 );

    infoHdr.size = 40;
    infoHdr.width = width;
    infoHdr.height = height;
    infoHdr.planes = 1;
    infoHdr.bitsPerPixel = 24;
    infoHdr.compression = 0;
    infoHdr.imageSize = hdr.size - hdr.offset;
    infoHdr.xPelsPerMeter = 0;
    infoHdr.yPelsPerMeter = 0;
    infoHdr.clrImportant = 0;
    infoHdr.clrUsed = 0;

    FILE *fd;
    if(! (fd=fopen(name,"wb"))){
        printf("***** fail to open dest file *****\n");
        return;
    }

    fwrite(&hdr , sizeof(hdr) , 1 , fd);
    fwrite(&infoHdr , sizeof(infoHdr) , 1 , fd);

    int x,y;
    for(y = 0; y < infoHdr.height ; y++){
        for(x = 0 ; x < infoHdr.width ; x++){
            fputc( dst[ y * infoHdr.width + x ].z , fd);
            fputc( dst[ y * infoHdr.width + x ].y , fd);
            fputc( dst[ y * infoHdr.width + x ].x , fd);
            // printf("save one pixel\n");

        }

        //pass filling bytes
        for(x = 0 ; x < (4 - (3*infoHdr.width)%4)%4 ; x++ ){
            fputc(0 , fd);
        }
    }

    printf("image file writed over\n");
    fclose(fd);
}


__global__ void genGauss(float *gauss , int size , float sig ){

    __shared__ float sum;
    sum=0;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int mid = size/2 ;
    float sigma = sig;
    if(x<size && y<size){
        gauss[y*size + x] = __expf(-0.5*((x - mid)*(x - mid)+(y - mid)*(y - mid)) / (sigma*sigma)) / (2 * 3.1415926 * sigma *sigma);
        atomicAdd(&sum , gauss[y*size + x]);
    }
    __syncthreads();
    if(x<size && y<size){
        gauss[y*size + x] = gauss[y*size + x] /sum;
    }
}
__device__ float cannyRgb2grayscale(uchar3 pix){
    return (pix.x*0.299 + pix.y*0.587 + pix.z*0.114);
}

__global__ void rgb2grayscale(uchar3 * pix , int width , int height , float * ping){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if(x < width && y < height){
        ping[index] = pix[index].x*0.299 + pix[index].y*0.587 + pix[index].z*0.114;
    }
}

__global__ void gaussBlur(float * ping, float * pang , int width , int height , float * gauss , int gaussSize){
    __shared__ float gau[GauSize][GauSize];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;
    //load guass matrix
    if(threadIdx.x<GauSize && threadIdx.y<GauSize){
        gau[threadIdx.y][threadIdx.x] = gauss[ threadIdx.y * GauSize + threadIdx.x];
    }
    __syncthreads();
    //blur
    if(x<width && y<height ){
        float blurValue = 0.0f;
        int mid = GauSize/2 +1; 
        for(int i = mid-GauSize ; i <= GauSize-mid; i++){
            for(int j = mid-GauSize ; j <= GauSize-mid; j++){
                // blurValue += gau[i+mid-1][j+mid-1]*dst[(y+i)*width + x+j].x;
                if((i+y)<height && (i+y)>=0  &&  (j+x)<width && (j+x)>=0 )
                    blurValue += gau[i+mid-1][j+mid-1] * ping[index + i*width +j];
            }
        }
       

        // src[index].x = src[index].y = src[index].z = blurValue;
        pang[index] = blurValue;
    }
}

__global__ void calcGradient(float *pang,float *ping , float * dx , float *dy , int width , int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;
    if(x<width-1 && y< height-1){
        dx[index] = pang[index + 1] - pang[index];
        dy[index] = pang[index + width] - pang[index];
        //ping is gradient now
        ping[index] = sqrtf(dx[index]*dx[index]+dy[index]*dy[index] );
    }
    if(x<width && y< height && (x==width-1 || y==height-1)){
        ping[index] = 0;
    }
    
}

__global__ void nonMaxRestrain(float *pang, float *ping , float *dx , float *dy , int width , int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;
    if(x<width -2 && y<height -2   && x>0 &&y>0){

    if(pang[index] <= 0.0000000001f && pang[index] >= -0.0000000001f){
        ping[index] = 0.0f;
    }
    else{
        float dxAbs = abs(dx[index]);
        float dyAbs = abs(dy[index]);
        float weight = 0;
        float grad[4];
        if(dyAbs > dxAbs){
            weight = dxAbs/dyAbs;
            grad[1] = pang[index - width];
            grad[3] = pang[index + width];

            if(dx[index]*dy[index]>0){
                grad[0] = pang[index -width - 1];
                grad[2] = pang[index +width + 1];
            }
            else{
                grad[0] = pang[index -width + 1];
                grad[2] = pang[index +width - 1];
            }
        }else{
            weight = dyAbs / dxAbs;
            grad[1] = pang[index - 1];
            grad[3] = pang[index + 1];

            if(dx[index]*dy[index]>0){
                // grad[0] = pang[index +width - 1];
                // grad[2] = pang[index -width + 1];

                grad[0] = pang[index -width - 1];
                grad[2] = pang[index +width + 1];
            }
            else{
                // grad[0] = pang[index -width - 1];
                // grad[2] = pang[index +width + 1];

                grad[0] = pang[index -width + 1];
                grad[2] = pang[index +width - 1];
            }
        }
        //插值
        // float grad1 = weight * grad[0] + (1-weight) * grad[1];
        // float grad2 = weight * grad[2] + (1-weight) * grad[3];

        float grad1 = weight * grad[1] + (1-weight) * grad[0];
        float grad2 = weight * grad[3] + (1-weight) * grad[2];
        if(pang[index] > grad1 && pang[index] > grad2){
            ping[index] = pang[index];
        } else{
            ping[index] = 0.0f;
        }
    }

    }//x<width y<height
    // else{
    //     ping[index] = 0;
    // }

}

__global__ void findMaxGrad(float * ping , int width , int height , float * max){
    extern __shared__ float temp[];

    int tid = threadIdx.x;
    int idx = tid + blockDim.x * blockIdx.x;

    if(idx<width*height){
        temp[tid] = ping[idx];
    }else{
        temp[tid] = 0;
    }

    int prevSize = blockDim.x;
    for(int d = blockDim.x >>1 ; d>0; d >>= 1){
        __syncthreads();
        if(tid<d  ){
            temp[tid] = temp[tid]>temp[tid+d]?temp[tid]:temp[tid + d];
            //for arbitrary block size
            if(prevSize > d*2 && (tid+2*d)<prevSize   ){
                temp[tid] = temp[tid]>temp[tid+2*d]?temp[tid]:temp[tid + 2*d];

            }
            prevSize = d;
        }
    }

    if(tid==0)max[blockIdx.x] = temp[0];
}

__global__ void edgeTrace(float *ping , float *pang , float *maxptr , int width , int height , int iteration , float hsigma , float lsigma){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;
    float max = *maxptr;
    float lhd = max * lsigma;
    float hhd = max * hsigma;

    for(int i =0 ; i<iteration ;i++){
    __syncthreads();
    if(x<width -1 && y<height -1   && x>0 && y>0){

        if(ping[index] < lhd)pang[index] = 0;
        else if(ping[index]>= hhd)pang[index]=1;
        else {
            if(ping[index-width-1]>=hhd || ping[index-width]>=hhd ||ping[index-width+1]>=hhd 
            || ping[index-1]>=hhd || ping[index+1]>=hhd
            || ping[index+width-1]>=hhd || ping[index+width]>=hhd ||ping[index+width+1]>=hhd){
                pang[index] = 1; 
                ping[index] = hhd;
            }else{
                pang[index] = 0;
                ping[index] = hhd;
            }
        }

    }//x, y
    }
}

__global__ void edgeTraceOptimized(float *ping , float *pang , float *maxptr , int width , int height , int iteration , float hsigma , float lsigma){
    __shared__ float tempPing[34][34] ;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;
    float max = *maxptr;
    float lhd = max * lsigma;
    float hhd = max * hsigma;

    int tx = threadIdx.x+1;
    int ty = threadIdx.y+1;

    if(x<width && y < height)tempPing[threadIdx.y +1][threadIdx.x +1] = ping[index];
    if(threadIdx.x ==0 ){
        if(x>0 && x<width && y<height){
            tempPing[threadIdx.y +1][0] = ping[index - 1];
        }
        else{
            tempPing[threadIdx.y +1][0] = 0;
        } 
    }
    if(threadIdx.y ==0){
        if(y>0 && x<width && y<height){
            tempPing[0][threadIdx.x +1] = ping[index - width];
        }
        else {
            tempPing[0][threadIdx.x +1] = 0;
        }
    }
    if(threadIdx.x == 31){
        if(x<width-1 && y<height){
            tempPing[threadIdx.y +1][33] = ping[index+1];
        }
        else {
            tempPing[threadIdx.y +1][33] = 0;

        }
    }
    if(threadIdx.y ==31){
        if(y<height-1 && x<width){
            tempPing[33][threadIdx.x +1] = ping[index + width];
        }
        else{
            tempPing[33][threadIdx.x +1] = 0;
        }
    }

    if(threadIdx.x == 0 && threadIdx.y == 0){
        if(x>0 && y >0  && x<width && y<height)tempPing[0][0] = ping[index -width -1];
        else tempPing[0][0] = 0;
    }
    if(threadIdx.x == 31 && threadIdx.y == 0){
        if(y>0 && x < width-1  && y <height)tempPing[0][33] = ping[index - width +1];
        else tempPing[0][33] = 0;
    }
    if(threadIdx.x == 31 && threadIdx.y == 31){
        if(y<height-1 && x<width-1)tempPing[33][33] = ping[index + width +1];
        else tempPing[33][33] = 0;
    }
    if(threadIdx.x == 0 && threadIdx.y == 31){
        if(x>0 && y<height-1 && x<width)tempPing[33][0] = ping[index + width -1];
        else tempPing[33][0] = 0;
    }
    __syncthreads();

    for(int i =0 ; i<iteration ;i++){
    __syncthreads();

    if(x<width  && y<height){

        if(tempPing[ty][tx] < lhd)pang[index] = 0;
        else if(tempPing[ty][tx]>= hhd)pang[index]=1;
        else {
            if(tempPing[ty-1][tx-1]>=hhd || tempPing[ty-1][tx]>=hhd ||tempPing[ty-1][tx+1]>=hhd 
            || tempPing[ty][tx-1]>=hhd || tempPing[ty][tx+1]>=hhd
            || tempPing[ty+1][tx-1]>=hhd || tempPing[ty+1][tx]>=hhd ||tempPing[ty+1][tx+1]>=hhd){
                pang[index] = 1; 
                tempPing[ty][tx] = hhd;
            }else{
                pang[index] = 0;
                tempPing[ty][tx]= hhd;
            }
        }

    }//x, y
    }
}

__global__ void trans2Bmp(float *ping , uchar3* imagedata , int width ,int height , int rev){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if(x< width && y < height){
        // imagedata[index].x = imagedata[index].y = imagedata[index].z = (ping[index]<25?0:255);
        if(rev)imagedata[index].x = imagedata[index].y = imagedata[index].z = ping[index]*255;
        else imagedata[index].x = imagedata[index].y = imagedata[index].z =(1- ping[index])*255;
        // imagedata[index].x = imagedata[index].y = imagedata[index].z = ping[index];
    }
}

 
 void cookargument(int argc , char* argv[] , float *gaussSigma , float *hsigma , float *lsigma ,int *iteration , char** srcName , char** dstName , int *opti , int *rev){
    //init
    *gaussSigma = 1.2; //-gs
    *hsigma = 0.2;     //-hs
    *lsigma = 0.1;     //-ls
    *iteration = 10;   //-it
    *opti =1;          //-opt
    *rev = 1;          //-rev
                       //-src
                       //-dst
    //no argument
    if(argc ==1)return ;

    int status;

    for(int i=1 ; i<argc;i++){
        //set gaussSigma
        if(strcmp("-gs",argv[i])==0){
            if(i<(argc-1)){
                i++;
                status = sscanf(argv[i] , "%f" , gaussSigma);
                if(status == 0 || status == EOF){
                    printf("*****invalid argument format*****\n");
                    exit(1);
                }
            }
            else{
                printf("*****invalid argument format*****\n");
                exit(1);
            }
        }
        //set hsigma
        if(strcmp("-hs" , argv[i]) == 0){
            if(i<(argc-1)){
                i++;
                status = sscanf(argv[i] , "%f" , hsigma);
                if(status == 0 || status == EOF){
                    printf("*****invalid argument format*****\n");
                    exit(1);
                }
            }else{
                printf("*****invalid argument format*****\n");
                exit(1);
            }
        }
        //set lsigma
        if(strcmp("-ls" , argv[i]) == 0){
            if(i<(argc-1)){
                i++;
                status = sscanf(argv[i] , "%f" , lsigma);
                if(status == 0 || status == EOF){
                    printf("*****invalid argument format*****\n");
                    exit(1);
                }
            }else{
                printf("*****invalid argument format*****\n");
                exit(1);
            }
        }
        //set iteration
        if(strcmp("-it" , argv[i]) == 0){
            if(i<(argc-1)){
                i++;
                status = sscanf(argv[i] , "%d" , iteration);
                if(status == 0 || status == EOF){
                    printf("*****invalid argument format*****\n");
                    exit(1);
                }
            }else{
                printf("*****invalid argument format*****\n");
                exit(1);
            }
        }
        //set src file name
        if(strcmp("-src" , argv[i]) == 0){
            if(i<(argc-1)){
                i++;
                *srcName = argv[i];
            }else{
                printf("*****invalid argument format*****\n");
                exit(1);
            }
        }
        //set dst file name
        if(strcmp("-dst" , argv[i]) == 0){
            if(i<(argc-1)){
                i++;
                *dstName = argv[i];
            }else{
                printf("*****invalid argument format*****\n");
                exit(1);
            }
        }
        //set opotimized or not
        if(strcmp("-opt" , argv[i]) == 0){
            if(i<(argc-1)){
                i++;
                status = sscanf(argv[i] , "%d" , opti);
                if(status == 0 || status == EOF){
                    printf("*****invalid argument format*****\n");
                    exit(1);
                }
            }else{
                printf("*****invalid argument format*****\n");
                exit(1);
            }
        }
        //set edge color 1 white 0 black
        if(strcmp("-rev" , argv[i]) == 0){
            if(i<(argc-1)){
                i++;
                status = sscanf(argv[i] , "%d" , rev);
                if(status == 0 || status == EOF){
                    printf("*****invalid argument format*****\n");
                    exit(1);
                }
            }else{
                printf("*****invalid argument format*****\n");
                exit(1);
            }
        }
        //print help
        if(strcmp("-help" , argv[i]) == 0 ){
            printf("-src srcname : set src image file, must be 24 bit bmp file\n");
            printf("-dst dstname : set dst image file, must be 24 bit bmp file\n");
            printf("-gs  gaussSigma: set gaussSigma\n");
            printf("-ls  lowThreshold : set lowthreshold in canny\n");
            printf("-hs  highThreshold: set highthreshold in canny\n");
            printf("-it  iteration    : set iteration number of tracing edge\n");
            printf("-opt  1/0    : set ioptimimze tracing edge or not\n");
            printf("-rev  1/0    : set  edge color black(0)/white(1)\n");
            printf("-help :show info like this\n");
            
            exit(0);
        }

        
    }//for
    
    //test print
    // printf("gausssigma:%f\n", *gaussSigma);
    // printf("lsigma:%f\n", *lsigma);
    // printf("hsigma:%f\n", *hsigma);
    // printf("iteration:%d\n", *iteration);


    //check value
    if(*iteration <1){
        printf("*****invalid iteration*****\n");
        exit(1);
    }
    if(*lsigma > *hsigma || *lsigma < 0 || *hsigma >1){
        printf("*****invalid l/hsigma value*****\n");
        exit(1);
    }
    if(*gaussSigma<=0){
        printf("*****invalid gauss sigma value*****\n");
        exit(1);
    }
 }


int main(int argc , char* argv[]){
    clock_t start , end;
    start = clock();
    clock_t startCanny , endCanny;

    uchar3 *imagedataSrc , *imagedataDst;
    uchar3 *d_imagedataSrc ,*d_imagedataDst;
    float *d_gauss;
    float * d_ping, *d_pang;
    float *d_dx , *d_dy;
    float *d_max;
    // float gauss[GauSize][GauSize];
    int gaussSize = GauSize;
    int width , height;

    float gaussSigma =1.2;
    float hsigma = 0.15;
    float lsigma = 0.1;
    int iteration = 10;
    int optimized = 1;
    int  rev = 1;

    // const char srcName[] = "../data/src3.bmp";
    // const char dstName[] = "../data/out.bmp";
    char *srcName = NULL ;
    char *dstName = NULL ;

    cookargument(argc , argv , &gaussSigma , &hsigma , &lsigma ,&iteration , &srcName , &dstName , &optimized,&rev);    
    if(srcName ==NULL)srcName = "../data/src3.bmp";
    if(dstName ==NULL)dstName = "../data/out.bmp";
    //load bmp format image file
    loadBMPFile(&imagedataSrc , &width , &height , srcName );
    // height /= 4;
    cudaError_t cudaStatus = cudaMalloc((void**)&d_gauss , gaussSize*gaussSize*sizeof(float));
    if(cudaStatus!=cudaSuccess){
        printf("*****fail to malloc gpu memory*****\n");
        return 1;
    }
    // cudaMalloc(&d_imagedataDst , width*height*sizeof(uchar3));
    cudaStatus = cudaMalloc(&d_ping , width*height*sizeof(float));
    if(cudaStatus!=cudaSuccess){
        printf("*****fail to malloc gpu memory*****\n");
        return 1;
    }
    cudaStatus = cudaMalloc(&d_pang , width*height*sizeof(float));
    if(cudaStatus!=cudaSuccess){
        printf("*****fail to malloc gpu memory*****\n");
        return 1;
    }
    cudaStatus = cudaMalloc(&d_dx , width*height*sizeof(float));
    if(cudaStatus!=cudaSuccess){
        printf("*****fail to malloc gpu memory*****\n");
        return 1;
    }
    cudaStatus = cudaMalloc(&d_dy , width*height*sizeof(float));
    if(cudaStatus!=cudaSuccess){
        printf("*****fail to malloc gpu memory*****\n");
        return 1;
    }

    cudaStatus = cudaMalloc(&d_imagedataSrc , width*height*sizeof(uchar3));
    if(cudaStatus!=cudaSuccess){
        printf("*****fail to malloc gpu memory*****\n");
        return 1;
    }
    // imagedataDst = (uchar3*)malloc( width*height*sizeof(uchar3));

    
    cudaStatus = cudaMemcpy(d_imagedataSrc , imagedataSrc , width*height*sizeof(uchar3) , cudaMemcpyHostToDevice );
    if(cudaStatus!=cudaSuccess){
        printf("*****fail to copy memory*****\n");
        return 1;
    }
    dim3 numofBlock(32 , 32 , 1);
    dim3 threadsPerBlock((width+31)/32 , (height+31)/32, 1);
    int threadnum = numofBlock.x*numofBlock.y*numofBlock.z;
    int blocknum = (width*height+threadnum-1)/threadnum;
    cudaMalloc(&d_max , blocknum*sizeof(float));

    //main process in gpu
    startCanny = clock();
    //generate gaussian filter
    genGauss<<<1 ,numofBlock >>>(d_gauss , gaussSize ,gaussSigma);
    // cudaDeviceSynchronize();
    
    //calc image's grayscale value
    rgb2grayscale<<< threadsPerBlock, numofBlock>>>(d_imagedataSrc , width , height , d_ping );
    // cudaDeviceSynchronize();
    
    //use gaussian filter
    gaussBlur<<< threadsPerBlock, numofBlock>>>(d_ping , d_pang , width , height , d_gauss , gaussSize);
    // cudaDeviceSynchronize();
    
    //calc gradient of every pixel    
    calcGradient<<< threadsPerBlock, numofBlock>>>(d_pang ,d_ping , d_dx , d_dy , width , height);
    //non max gradient restrain
    nonMaxRestrain<<< threadsPerBlock, numofBlock>>>(d_ping , d_pang , d_dx , d_dy , width , height);
    //find max gradient of the whole image
    //local max    
    findMaxGrad<<<blocknum,threadnum  , threadnum*sizeof(float)>>>(d_pang , width , height , d_max);
    //global max
    findMaxGrad<<<1, blocknum , blocknum*sizeof(float)>>>(d_max , blocknum , 1 , d_max);
    // cudaDeviceSynchronize();
    // float max;
    // cudaMemcpy(&max , d_max , sizeof(float) , cudaMemcpyDeviceToHost );
    // printf("max is %f\n",max);
    // printf("blocknum is %d\n" , blocknum);
    
    //Tracing edges through the image and hysteresis thresholding
    if(!optimized)edgeTrace<<<threadsPerBlock, numofBlock>>>(d_pang ,d_ping , d_max , width , height , iteration , hsigma , lsigma);
    else edgeTraceOptimized<<<threadsPerBlock, numofBlock>>>(d_pang ,d_ping , d_max , width , height , iteration , hsigma , lsigma);
    
    //translate bit value to bmp file format
    trans2Bmp<<< threadsPerBlock, numofBlock>>>(d_ping , d_imagedataSrc , width , height,rev);
    //main process end

    cudaStatus = cudaDeviceSynchronize();
    if(cudaStatus != cudaSuccess){
        printf("*****kernel function down*****\n");
    }
    endCanny = clock();
    printf("------canny use %ld ms\n" , endCanny - startCanny);

    cudaStatus = cudaMemcpy(imagedataSrc , d_imagedataSrc , width*height*sizeof(uchar3) , cudaMemcpyDeviceToHost );
    if(cudaStatus!=cudaSuccess){
        printf("*****fail to copy memory from device*****\n");
        return 1;
    }

    // for(int i=0 ; i<width*height/1024;i++){
    // cudaStatus = cudaMemcpy(imagedataSrc+i*1024 , d_imagedataSrc+i*1024 , 1024*sizeof(uchar3) , cudaMemcpyDeviceToHost );
    // if(cudaStatus!=cudaSuccess){
    //     printf("*****fail to copy memory from device*****\n");
    //     return 1;
    // }
    // }
    // if((width*height)%1024!=0){
    //     cudaStatus = cudaMemcpy(imagedataSrc+i*1024 , d_imagedataSrc+i*1024 , ((width*height)%1024)*sizeof(uchar3) , cudaMemcpyDeviceToHost );
    //     if(cudaStatus!=cudaSuccess){
    //         printf("*****fail to copy memory from device*****\n");
    //         return 1;
    //     }  
    // }

    printf("before save bmlp file\n");
    printf("width:%d height:%d\n", width , height);
    saveBMPFile(imagedataSrc ,width , height , dstName);

    // cudaFree(d_imagedataDst);
    cudaFree(d_imagedataSrc);
    cudaFree(d_gauss);
    cudaFree(d_ping);
    cudaFree(d_pang);
    cudaFree(d_dx);
    cudaFree(d_dy);
    cudaFree(d_max);
    // free(imagedataDst);
    free(imagedataSrc);

    end =  clock();
    printf("------total use %ld ms\n" , end - start);
    return 0;
}

