#ifndef IMG_UTI_H_
#define IMG_UTI_H_

#include <opencv2/opencv.hpp>

namespace img_uti{
       cv::Mat CropResize(cv::Mat& input_image, int image_size);
}


#endif
