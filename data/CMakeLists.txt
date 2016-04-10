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


# add_custom_target(clean-cmake-files
#    COMMAND ${CMAKE_COMMAND} -P clean-all.cmake
# )
# 
# # clean-all.cmake
# set(cmake_generated ${CMAKE_BINARY_DIR}/CMakeCache.txt
#                     ${CMAKE_BINARY_DIR}/cmake_install.cmake  
#                     ${CMAKE_BINARY_DIR}/Makefile
#                     ${CMAKE_BINARY_DIR}/CMakeFiles
# )
# 
# foreach(file ${cmake_generated})
# 
#   if (EXISTS ${file})
#      file(REMOVE_RECURSE ${file})
#   endif()
# 
# endforeach(file)
# 
# add_custom_target(clean-all
#    COMMAND ${CMAKE_BUILD_TOOL} clean
#    COMMAND ${CMAKE_COMMAND} -P clean-all.cmake
# )

# 
# foreach (src ${SRC_LIST})
#     set(NAME, "${src}.bin")
#     add_executable(${NAME} ${src})
#     target_link_libraries( ${NAME} ${ALL_LIB})
# endforeach(src ${SRC_LIST})
# 
# add_executable( AnnotVideo annotate_video.cpp ${annot_cc})
# target_link_libraries( AnnotVideo ${ALL_LIB})
# 
# project( CleanImage)
# add_executable( CleanImage clean_image.cpp ${annot_cc})
# target_link_libraries( CleanImage ${ALL_LIB})
# 
# project(DeleteImage)
# add_executable( DeleteImage delete_image.cpp ${annot_cc})
# target_link_libraries( DeleteImage ${ALL_LIB})
# 
# project(DeleteNonImage)
# add_executable( DeleteNonImage delete_non_image.cpp ${annot_cc})
# target_link_libraries( DeleteNonImage ${ALL_LIB})
