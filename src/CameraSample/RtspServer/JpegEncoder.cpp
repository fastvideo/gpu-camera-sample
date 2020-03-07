#include "JpegEncoder.h"

#include <iostream>

#if 0
#include "turbojpeg.h"
#else
#include "jpeglib.h"
#endif

jpeg_encoder::jpeg_encoder()
{

}

jpeg_encoder::~jpeg_encoder()
{

}

#if 0
bool jpeg_encoder::encode(unsigned char *input, int width, int height, std::vector<uchar> &output, int quality)
{
    tjhandle handle = tjInitCompress();

    int pixFormat = TJPF_GRAY;
    int outSubSamp = TJSAMP_GRAY;

    int flags = TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT | TJFLAG_FORCESSE;

    pixFormat = TJPF_RGB;
    outSubSamp = TJSAMP_420;

    uchar *jpegBuf = nullptr;
    ulong jpegSize = 0;

    int pitch = static_cast<int>(width * 3);
    if(tjCompress2(handle, input, width, pitch, height,
            pixFormat, &jpegBuf, &jpegSize, outSubSamp, quality, flags) < 0){
        return false;
    }

    output.resize(jpegSize);
    std::copy(jpegBuf, jpegBuf + jpegSize, output.data());

    tjFree(jpegBuf);

    return true;
}

bool jpeg_encoder::encode(unsigned char *input, int width, int height, uchar *output, uint &size, int quality)
{
    tjhandle handle = tjInitCompress();

    int pixFormat = TJPF_GRAY;
    int outSubSamp = TJSAMP_GRAY;

    int flags = TJFLAG_ACCURATEDCT | TJFLAG_PROGRESSIVE;

    pixFormat = TJPF_RGB;
    outSubSamp = TJSAMP_420;

    uchar *jpegBuf = nullptr;
    ulong jpegSize = 0;

    int pitch = width * 3;
    if(tjCompress2(handle, input, width, pitch, height,
            pixFormat, &jpegBuf, &jpegSize, outSubSamp, quality, flags) < 0){
        return false;
    }

    std::copy(jpegBuf, jpegBuf + jpegSize, output);
    size = jpegSize;

    tjFree(jpegBuf);

    tjDestroy(handle);

    return true;
}
#else

#define OUTPUT_BUF_SIZE  4096

typedef struct {
    struct jpeg_destination_mgr pub; /* public fields */

    std::vector<uchar> *output;
    std::vector<uchar> *buffer;		/* start of buffer */
} my_destination_mgr;

typedef my_destination_mgr * my_dest_ptr;

METHODDEF(void)
init_destination (j_compress_ptr cinfo)
{
    my_dest_ptr dest = (my_dest_ptr) cinfo->dest;

    dest->buffer->clear();
    dest->output->clear();

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

////////////

void error_exit(j_common_ptr cinfo)
{
    std::cout << "Error " << cinfo->err->msg_code << " " << cinfo->err->msg_parm.i[0] << std::endl;
}

bool jpeg_encoder::encode(unsigned char *input, int width, int height,
                          int channels, std::vector<uchar> &output, int quality)
{
    jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    cinfo.err = jpeg_std_error(&jerr);
    cinfo.err->error_exit = error_exit;

    cinfo.in_color_space = JCS_RGB;
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.jpeg_color_space = JCS_YCbCr;

    my_destination_mgr dest;

//    if (cinfo.dest == nullptr) {	/* first time for this JPEG object? */
//        cinfo.dest = (struct jpeg_destination_mgr *)
//                (*cinfo.mem->alloc_small) ((j_common_ptr) &cinfo, JPOOL_PERMANENT,
//                                           sizeof(my_destination_mgr));
//    }

    cinfo.dest = (struct jpeg_destination_mgr *)&dest;

    dest.pub.init_destination = init_destination;
    dest.pub.empty_output_buffer = empty_dst;
    dest.pub.term_destination = term_destination;

    std::vector<uchar> buffer;
    dest.buffer = &buffer;
    dest.output = &output;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);
    jpeg_start_compress(&cinfo, true);

    JSAMPROW arr[1];
    int pitch = width * 3;

    if(channels == 1){
        int pitchIn = width;

        std::vector<unsigned char> row;
        row.resize(pitch);
        arr[0] = row.data();

        while(cinfo.next_scanline < cinfo.image_height){
            unsigned char *in = input + cinfo.next_scanline * pitchIn;
            for(int i = 0; i < width; ++i){
                row.data()[i * 3 + 0] = in[i];
                row.data()[i * 3 + 1] = in[i];
                row.data()[i * 3 + 2] = in[i];
            }

            jpeg_write_scanlines(&cinfo, arr, 1);
        }
    }else{
        while(cinfo.next_scanline < cinfo.image_height){
            arr[0] = input + cinfo.next_scanline * pitch;
            jpeg_write_scanlines(&cinfo, arr, 1);
        }
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    return true;
}

bool jpeg_encoder::encode(unsigned char *input, int width, int height,
                          int channels, uchar *output, uint &size, int quality)
{
    input, width, height, channels, output, size, quality;
    return false;
}

#endif
