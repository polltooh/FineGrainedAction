cmake_minimum_required(VERSION 2.8)

project( AnnotVideo )
find_package( OpenCV REQUIRED )
find_package( Boost COMPONENTS system filesystem REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

file(GLOB annot_cc
    "*.cc"
)

add_executable( AnnotVideo annotate_video.cpp
                            ${annot_cc})

target_link_libraries( AnnotVideo
                        ${OpenCV_LIBS}
                        ${Boost_FILESYSTEM_LIBRARY} 
                        ${Boost_SYSTEM_LIBRARY}
                     )

project( CleanImage)

add_executable( CleanImage clean_image.cpp
                            ${annot_cc})

target_link_libraries( CleanImage
                        ${OpenCV_LIBS}
                        ${Boost_FILESYSTEM_LIBRARY} 
                        ${Boost_SYSTEM_LIBRARY}
                     )
