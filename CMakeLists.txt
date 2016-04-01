cmake_minimum_required(VERSION 2.8)

project( AnnotVideo )
find_package( OpenCV REQUIRED )

file(GLOB annot_cpp
    "*.cpp"
)

add_executable( AnnotVideo ${annot_cpp})
target_link_libraries( AnnotVideo ${OpenCV_LIBS} )
