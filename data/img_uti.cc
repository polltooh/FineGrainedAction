#include "img_uti.h"


cv::Mat img_uti::CropResize(cv::Mat& input_image, int image_size){

int org_row = input_image.rows;
int org_col = input_image.cols;
cv::Mat croped_image;
cv::Mat resized_image;
cv::Rect crop_reg;

if (org_row <= org_col){
    int start_col = (org_col - org_row)/2;
    int end_col = start_col + org_row;
    crop_reg = cv::Rect(start_col, 0, org_row, org_row);
}
else{
    int start_row = (org_row - org_col)/2;
    int end_row = start_row + org_col;
    crop_reg = cv::Rect(0, start_row, org_col, org_col);
}

croped_image = input_image(crop_reg);
cv::resize(croped_image, resized_image, cv::Size(image_size, image_size));

return resized_image;

}
