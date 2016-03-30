cmake_minimum_required(VERSION 2.8)

project( AnnotVideo )
find_package( OpenCV REQUIRED )

add_executable( AnnotVideo annotate_video.cpp utility_function.cpp)

target_link_libraries( AnnotVideo ${OpenCV_LIBS} )
