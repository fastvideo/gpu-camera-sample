#include "jpegenc.h"


#include <stdio.h>
#include <iostream>

#include "jpeglib.h"

#ifdef _MSC_VER
#include "turbojpeg.h"
#endif

//////////////////////////////////
//////////////////////////////////

void error_exit(j_common_ptr cinfo)
{
    std::cout << "Error " << cinfo->err->msg_code << " " << cinfo->err->msg_parm.i[0] << std::endl;
}

///////////////////////////////////

#define OUTPUT_BUF_SIZE  4096

typedef struct {
    struct jpeg_destination_mgr pub; /* public fields */

    bytearray *output;
    bytearray *buffer;		/* start of buffer */
} my_destination_mgr;

typedef my_destination_mgr * my_dest_ptr;

METHODDEF(void)
init_destination (j_compress_ptr cinfo)
{
    my_dest_ptr dest = (my_dest_ptr) cinfo->dest;

    /* Allocate the output buffer --- it will be released when done with image */
    dest->buffer->resize(OUTPUT_BUF_SIZE);

    dest->pub.next_output_byte = (JOCTET*)dest->buffer->data();
    dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;
}


METHODDEF(boolean)
empty_dst (j_compress_ptr cinfo)
{
    my_dest_ptr dest = (my_dest_ptr) cinfo->dest;

    size_t size = dest->output->size();
    size_t size_add = dest->buffer->size();
    dest->output->resize(size + size_add);
    std::copy(dest->buffer->data(), dest->buffer->data() + size_add, dest->output->data() + size);//append(*dest->buffer);

    dest->pub.next_output_byte = (JOCTET*)dest->buffer->data();
    dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;

    return TRUE;
}


/*
 * Terminate destination --- called by jpeg_finish_compress
 * after all data has been written.  Usually needs to flush buffer.
 *
 * NB: *not* called by jpeg_abort or jpeg_destroy; surrounding
 * application must deal with any cleanup that should happen even
 * for error exit.
 */

METHODDEF(void)
term_destination (j_compress_ptr cinfo)
{
    my_dest_ptr dest = (my_dest_ptr) cinfo->dest;
    size_t datacount = OUTPUT_BUF_SIZE - dest->pub.free_in_buffer;

    /* Write any data remaining in the buffer */
    if (datacount > 0) {
        size_t size = dest->output->size();
        size_t size_add = datacount;
        dest->output->resize(size + size_add);
        std::copy(dest->buffer->data(), dest->buffer->data() + size_add, dest->output->data() + size);//append(*dest->buffer);
        //dest->output->insert(dest->buffer->data(), datacount);
    }
}

////////

jpegenc::jpegenc() {

}

bool jpegenc::encode(const bytearray input[3], int width, int height, bytearray& output, int quality) {

    if(input[0].empty())
        return false;
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    int row_stride1, row_stride2;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    cinfo.err = jpeg_std_error(&jerr);
    cinfo.err->error_exit = error_exit;

    my_dest_ptr dest;

    if (cinfo.dest == nullptr) {	/* first time for this JPEG object? */
        cinfo.dest = (struct jpeg_destination_mgr *)
                (*cinfo.mem->alloc_small) ((j_common_ptr) &cinfo, JPOOL_PERMANENT,
                                           sizeof(my_destination_mgr));
    }

    dest = (my_dest_ptr) cinfo.dest;
    dest->pub.init_destination = init_destination;
    dest->pub.empty_output_buffer = empty_dst;
    dest->pub.term_destination = term_destination;

    bytearray buffer;
    dest->buffer = &buffer;
    dest->output = &output;

    cinfo.image_width = width; 	/* image width and height, in pixels */
    cinfo.image_height = height;
	cinfo.input_components = 3;		/* # of color components per pixel */
    cinfo.in_color_space = JCS_YCbCr; 	/* colorspace of input image */

    cinfo.jpeg_color_space = JCS_YCbCr;
//	QByteArray lines[3];
//	cnv_rgb2yuv422(input, lines);

    jpeg_set_defaults(&cinfo);
    //cinfo.dct_method = JDCT_IFAST;
    /* Now you can set any non-default parameters you wish to.
     * Here we just illustrate the use of quality (quantization table) scaling:
     */
    jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);

    jpeg_start_compress(&cinfo, true);

    int type = 3;

	JSAMPROW row_pointer[1];
    row_stride1 = width;	/* JSAMPLEs per row in image_buffer */
    row_stride2 = width/2;

    if(type == 3){

		bytearray row;
		row.resize(row_stride1 * 3);

        int cur = 0;
        while (cinfo.next_scanline < cinfo.image_height) {

            const unsigned char* d1 = input[0].data() + cur * row_stride1;
            const unsigned char* d2 = input[1].data() + cur * row_stride2;
            const unsigned char* d3 = input[2].data() + cur * row_stride2;
			for(int i = 0; i < row_stride2; ++i){
				row[i * 6 + 0] = d1[i * 2 + 0];
				row[i * 6 + 1] = d2[i];
				row[i * 6 + 2] = d3[i];
				row[i * 6 + 3] = d1[i * 2 + 1];
				row[i * 6 + 4] = d2[i];
				row[i * 6 + 5] = d3[i];
			}

			row_pointer[0] = (JSAMPROW)row.data();
			(void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
            cur++;
        }
    }
    /* Step 6: Finish compression */

    jpeg_finish_compress(&cinfo);

    /* Step 7: release JPEG compression object */

    /* This is an important step since it will release a good deal of memory. */
    jpeg_destroy_compress(&cinfo);

    return true;
}


//////////////////////////////

bool jpegenc::decode(const bytearray &input, PImage &image)
{
    if(input.empty())
        return false;

#ifdef _MSC_VER
    tjhandle handle = tjInitDecompress();
    int w, h, subSamp, cp;
    int res = tjDecompressHeader3(handle, input.data(), (unsigned)input.size(), &w, &h, &subSamp, &cp);
    if(res < 0){
        //char *str = tjGetErrorStr2(handle);
        //printf("error %s        \r", str);
        //tjDestroy(handle);
        //return false;
    }

    if(!image.get() || image->width != w || image->height != h){
        image.reset(new RTSPImage);
        image->setRGB(w, h);
    }
    int pixSize = cp == TJCS_GRAY? 1 : 3;
    int pitch = cp == w * pixSize;

    int flags = TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT | TJFLAG_FORCESSE;
    res = tjDecompress(handle, (unsigned char*)input.data(), (unsigned)input.size(), image->rgb.data(), w, pitch, h, pixSize, flags);

    tjDestroy(handle);

    return res == 0;
#else
    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;

    jpeg_create_decompress(&cinfo);

    cinfo.err = jpeg_std_error(&jerr);
    cinfo.err->error_exit = error_exit;

    cinfo.out_color_space = JCS_RGB;

    jpeg_mem_src(&cinfo, input.data(), input.size());

    jpeg_read_header(&cinfo, true);

    jpeg_start_decompress(&cinfo);

    JSAMPROW scanline;

    int w = cinfo.output_width;
    int h = cinfo.output_height;
    int numEl = 3;

    if(!image.get() || image->width != w || image->height != h){
        image.reset(new RTSPImage);
        image->setRGB(w, h);
    }

    int pitch = w * numEl;

//    jpeg_finish_decompress(&cinfo);
//    jpeg_destroy_decompress(&cinfo);
//    return true;

    while(cinfo.output_scanline < cinfo.output_height){
        scanline = image->rgb.data() + cinfo.output_scanline * pitch;
        int numscanlines = jpeg_read_scanlines(&cinfo, &scanline, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    return  true;
#endif
}

bool jpegenc::decode(const uint8_t *input, int len, PImage &image)
{
#ifdef _MSC_VER
    tjhandle handle = tjInitDecompress();
    int w, h, subSamp, cp;
    int res = tjDecompressHeader3(handle, input, len, &w, &h, &subSamp, &cp);
    if(res < 0){
        //char *str = tjGetErrorStr2(handle);
        //printf("error %s        \r", str);
        //tjDestroy(handle);
        //return false;
    }

    if(!image.get() || image->width != w || image->height != h){
        image.reset(new RTSPImage);
		if(cp == TJCS_GRAY){
			image->setGray(w, h);
		}else{
			image->setRGB(w, h);
		}
    }
    int pixSize = cp == TJCS_GRAY? 1 : 3;
    int pitch = cp == w * pixSize;

    int flags = TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT | TJFLAG_FORCESSE;
    res = tjDecompress(handle, (unsigned char*)input, len, image->rgb.data(), w, pitch, h, pixSize, flags);

    tjDestroy(handle);

    return res == 0;
#else
    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;

    jpeg_create_decompress(&cinfo);

    cinfo.err = jpeg_std_error(&jerr);
    cinfo.err->error_exit = error_exit;

    cinfo.out_color_space = JCS_RGB;

    jpeg_mem_src(&cinfo, input, len);

    jpeg_read_header(&cinfo, true);

    jpeg_start_decompress(&cinfo);

    JSAMPROW scanline;

    int w = cinfo.output_width;
    int h = cinfo.output_height;
    int numEl = 3;

    if(!image.get() || image->width != w || image->height != h){
        image.reset(new RTSPImage);
        image->setRGB(w, h);
    }

    int pitch = w * numEl;

//    jpeg_finish_decompress(&cinfo);
//    jpeg_destroy_decompress(&cinfo);
//    return true;

    while(cinfo.output_scanline < cinfo.output_height){
        scanline = image->rgb.data() + cinfo.output_scanline * pitch;
        int numscanlines = jpeg_read_scanlines(&cinfo, &scanline, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    return  true;
#endif
}
