#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include <math.h>

#include <time.h>

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

typedef struct {
    unsigned char x,y,z; //bmp中24位位图数据存储的单个像素通道顺序是bgr ， 且每行必须为4的倍数， 不够用0填充
}uchar3;

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


void genGauss(float *gauss , int size , float sig ){

    float sum=0;
    int mid = size/2 ;
    float sigma = sig;
    for(int y = 0;y<size;y++){
        for(int x =0 ;x < size ;x++){
            gauss[y*size + x] = expf(-0.5*((x - mid)*(x - mid)+(y - mid)*(y - mid)) / (sigma*sigma)) / (2 * 3.1415926 * sigma *sigma);
            sum += gauss[y*size + x];
        }
    }
    
    for(int y = 0; y<size ; y++){
        for(int x =0; x < size ; x++){
            gauss[y*size + x] = gauss[y*size + x] /sum;
        }
    }
    
}


void rgb2grayscale(uchar3 * pix , int width , int height , float * ping){
    
    for(int y=0 ; y<height ; y++){
        for(int x =0; x <width ;x++){
            int index = y * width + x;
            ping[index] = pix[index].x*0.299 + pix[index].y*0.587 + pix[index].z*0.114;
        }
    }
    
    
}

void gaussBlur(float * ping, float * pang , int width , int height , float * gauss , int gaussSize){
    for(int y=0 ; y<height ; y++){
        for(int x =0; x <width ;x++){
            int index = y * width + x;
    
            float blurValue = 0.0f;
            int mid = GauSize/2 +1; 
            for(int i = mid-GauSize ; i <= GauSize-mid; i++){
                for(int j = mid-GauSize ; j <= GauSize-mid; j++){
                    // blurValue += gau[i+mid-1][j+mid-1]*dst[(y+i)*width + x+j].x;
                    if((i+y)<height && (i+y)>=0  &&  (j+x)<width && (j+x)>=0 )
                        blurValue += gauss[(i+mid-1)*gaussSize +j+mid-1] * ping[index + i*width +j];
                }
            }
            

            // src[index].x = src[index].y = src[index].z = blurValue;
            pang[index] = blurValue;
        }
    }
    
    
}

void calcGradient(float *pang,float *ping , float * dx , float *dy , int width , int height){
    for(int y=0 ; y<height ; y++){
        for(int x =0; x <width ;x++){
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
    }
    
}

void nonMaxRestrain(float *pang, float *ping , float *dx , float *dy , int width , int height){
    for(int y=0 ; y<height ; y++){
        for(int x =0; x <width ;x++){

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
    }

}

void findMaxGrad(float * ping , int width , int height , float * max){
    float m=0;
    for(int i = 0; i<width * height ; i++){
        if(ping[i]>m)m = ping[i];
    }
    *max = m;
}

void edgeTrace(float *ping , float *pang , float *maxptr , int width , int height , int iteration , float hsigma , float lsigma){
   
    float max = *maxptr;
    float lhd = max * lsigma;
    float hhd = max * hsigma;

    for(int i =0 ; i<iteration ;i++){
        for(int y =1 ; y<height-1;y++){
            for(int x = 1; x<width-1 ;x++){
                int index = y * width + x;
                
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
            }
        }
    }
}


void trans2Bmp(float *ping , uchar3* imagedata , int width ,int height , int rev){
    for(int y=0 ; y<height ; y++){
        for(int x =0; x <width ;x++){
            int index = y * width + x;
        
            // imagedata[index].x = imagedata[index].y = imagedata[index].z = (ping[index]<25?0:255);
            if(rev)imagedata[index].x = imagedata[index].y = imagedata[index].z = ping[index]*255;
            else imagedata[index].x = imagedata[index].y = imagedata[index].z =(1- ping[index])*255;
            // imagedata[index].x = imagedata[index].y = imagedata[index].z = ping[index];
        }
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
    float *gauss;
    float * ping, *pang;
    float *dx , *dy;
    float max;
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
    gauss = (float*)malloc(gaussSize*gaussSize*sizeof(float));
    
    ping = (float*)malloc( width*height*sizeof(float));
    
    pang = (float*)malloc(width*height*sizeof(float));
   
    dx = (float*)malloc(width*height*sizeof(float));
   
    dy = (float*)malloc(width*height*sizeof(float));
    
    // imagedataDst = (uchar3*)malloc( width*height*sizeof(uchar3));

    //main process in cpu
    startCanny = clock();
    //generate gaussian filter
    genGauss(gauss , gaussSize ,gaussSigma);
    
    //calc image's grayscale value
    rgb2grayscale(imagedataSrc , width , height , ping );
    
    //use gaussian filter
    gaussBlur(ping , pang , width , height , gauss , gaussSize);
    
    //calc gradient of every pixel    
    calcGradient(pang ,ping , dx , dy , width , height);
    //non max gradient restrain
    nonMaxRestrain(ping , pang , dx , dy , width , height);
    //find max gradient of the whole image
    findMaxGrad(pang , width , height , &max);
   
    //Tracing edges through the image and hysteresis thresholding
    edgeTrace(pang ,ping , &max , width , height , iteration , hsigma , lsigma);
    
    //translate bit value to bmp file format
    trans2Bmp(ping , imagedataSrc , width , height,rev);
    //main process end
    endCanny = clock();
    printf("------canny use %ld ms\n" , endCanny - startCanny);

    printf("before save bmlp file\n");
    printf("width:%d height:%d\n", width , height);
    saveBMPFile(imagedataSrc ,width , height , dstName);

    free(gauss);
    free(ping);
    free(pang);
    free(dx);
    free(dy);
    // free(imagedataDst);
    free(imagedataSrc);

    end =  clock();
    printf("------total suse %ld ms\n" , end - start);
    return 0;
}

