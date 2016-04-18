cmake_minimum_required(VERSION 2.8)

project( AnnotVideo )
find_package( OpenCV REQUIRED )
find_package( Boost COMPONENTS system filesystem REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

file(GLOB C_FILE
    "*.cc"
)

file(GLOB SRC_LIST
    "*.cpp"
)

set(ALL_LIB "${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY}")

foreach (SRC ${SRC_LIST})
    get_filename_component(filename ${SRC} NAME_WE) 
    set(TARGET "${filename}.bin")
    message(${TARGET})
    project(${TARGET})
    add_executable(${TARGET} ${SRC} ${C_FILE})
    target_link_libraries( ${TARGET} ${ALL_LIB})
endforeach(SRC ${SRC_LIST})

