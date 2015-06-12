cmake_minimum_required(VERSION 2.8)
project(romeo_text_detection)

find_package(qibuild)

find_package(VISP REQUIRED)
#if(VISP_FOUND)
# include(${VISP_USE_FILE})
#endif()

find_package(visp_naoqi REQUIRED)

#include_directories( ${CMAKE_BINARY_DIR}/include)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(TESSERACT REQUIRED)
if(TESSERACT_FOUND)
  include_directories(${TESSERACT_INCLUDE_DIRS})
endif()



include_directories("${CMAKE_SOURCE_DIR}/lib")


#qi_create_lib(CCV
#  SRC
# ${CMAKE_SOURCE_DIR}/lib/ccv.h
#)
#qi_stage_lib(CCV)

# Create a executable named romeo_text_detection

qi_create_bin(detect_read_text detect_read_text.cpp ${CMAKE_SOURCE_DIR}/lib/ccv.h)
#qi_use_lib(detect_read_text  ${CMAKE_SOURCE_DIR}/lib/libccv.a)
qi_use_lib(detect_read_text  visp_naoqi ALCOMMON ALPROXIES ALVISION)
target_link_libraries(detect_read_text ${TESSERACT_LIBRARIES} ${CMAKE_SOURCE_DIR}/lib/libccv.a)

#add_executable(detect_read_text detect_read_text.cpp ${CMAKE_SOURCE_DIR}/lib/ccv.h)
#target_link_libraries(detect_read_text ${TESSERACT_LIBRARIES} ${CMAKE_SOURCE_DIR}/lib/libccv.a)
