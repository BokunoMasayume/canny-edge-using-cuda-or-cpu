#include <stdio.h>
#include <stdlib.h>

#pragma pack (1)
//设置对齐方式
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

void loadBMPFile(uchar3 ** dst, unsigned int *width , unsigned int *height , const char *name){

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
    if(infoHdr.compression){
        printf("***** cannot solve copressed image *****\n");
        return ;
    }


    *width = infoHdr.width;
    *height = infoHdr.height;
    //malloc space for image data 
    *dst = (uchar3 *)malloc(*width * *height * sizeof(uchar3));

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

void saveBMPFile(uchar3 *dst , unsigned int width , unsigned int height , const char *name){
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

        }

        //pass filling bytes
        for(x = 0 ; x < (4 - (3*infoHdr.width)%4)%4 ; x++ ){
            fputc(0 , fd);
        }
    }

    printf("image file writed over\n");
    fclose(fd);
}



int main(){

    uchar3 *imagedatalis;
    unsigned int width , height; 
    loadBMPFile(&imagedatalis ,&width , &height , "../data/src2.bmp" );

    for(int i =0 ; i<width*height ;i++){
        unsigned char r,g,b;
        r = imagedatalis[i].x;
        g = imagedatalis[i].y;
        b = imagedatalis[i].z;
        imagedatalis[i].x = imagedatalis[i].y = imagedatalis[i].z = r*0.299 + g*0.587 + b*0.114;

    }

    saveBMPFile(imagedatalis , width  ,height , "../data/out.bmp");

    return 0;
}





// void canny(float *gauss , int gaussSize , int width , int height ,uchar3 *src , float * ping, float * pang , float *dx, float *dy ){
//     __shared__ float gau[GauSize][GauSize];
    
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     int index = y * width + x;

//     //load guass matrix
//     if(threadIdx.x<GauSize && threadIdx.y<GauSize){
//         gau[threadIdx.y][threadIdx.x] = gauss[ threadIdx.y * GauSize + threadIdx.x];
//     }

//     //rgb2grayscale
//     __syncthreads();
//     if(x<width && y<height){  
//         // dst[index].x = dst[index].y = dst[index].z = cannyRgb2grayscale(src[index]);
//         ping[index] = cannyRgb2grayscale(src[index]);
//     }

//     //gauss blur
//     __syncthreads();
//     if(x<width && y<height ){
//         float blurValue = 0.0f;
//         int mid = GauSize/2 +1; 
//         for(int i = mid-gaussSize ; i <= gaussSize-mid; i++){
//             for(int j = mid-gaussSize ; j <= gaussSize-mid; j++){
//                 // blurValue += gau[i+mid-1][j+mid-1]*dst[(y+i)*width + x+j].x;
//                 blurValue += gau[i+mid-1][j+mid-1] * ping[index];
//             }
//         }
       

//         // src[index].x = src[index].y = src[index].z = blurValue;
//         pang[index] = blurValue;
//     }

//     //calc grandient, dx dy d saved in dst.x y z
//     __syncthreads();
//     dx[index] = pang[index + 1] - pang[index];
//     dy[index] = pang[index + width] - pang[index];
//     //pang is gradient now
//     pang[index] = sqrtf(dx[index]*dx[index]+dy[index]*dy[index] );

//     //非极大值抑制 NMS
//     __syncthreads();
//     if(pang[index] <= 0.0000000001f && pang[index] >= -0.0000000001f){
//         ping[index] = 0.0f;
//     }
//     else{
//         float dxAbs = abs(dx[index]);
//         float dyAbs = abs(dy[index]);
//         float weight = 0;
//         float grad[4];
//         if(dyAbs > dxAbs){
//             weight = dxAbs/dyAbs;
//             grad[1] = pang[index - width];
//             grad[3] = pang[index + width];

//             if(dx[index]*dy[index]>0){
//                 grad[0] = pang[index -width - 1];
//                 grad[2] = pang[index +width + 1];
//             }
//             else{
//                 grad[0] = pang[index -width + 1];
//                 grad[2] = pang[index +width - 1];
//             }
//         }else{
//             weight = dyAbs / dxAbs;
//             grad[1] = pang[index - 1];
//             grad[3] = pang[index + 1];

//             if(dx[index]*dy[index]>0){
//                 grad[0] = pang[index -width - 1];
//                 grad[2] = pang[index +width + 1];
//             }
//             else{
//                 grad[0] = pang[index -width + 1];
//                 grad[2] = pang[index +width - 1];
//             }
//         }
//         //插值
//         float grad1 = weight * grad[1] + (1-weight) * grad[0];
//         float grad2 = weight * grad[3] + (1-weight) * grad[2];
//         if(pang[index] > grad1 && pang[index] > grad2){
//             ping[index] = pang[index];
//         } else{
//             ping[index] = 0.0f;
//         }
//     }

//     //get max gradient value


//     //双阙值
//     // __syncthreads();
//     // float lhd = 0.2 *

//     // __syncthreads();
//     // if(x<width && y<height){
//     //     dst[index].x = dst[index].y = dst[index].z = src[index].x;
//     // }



    



// }