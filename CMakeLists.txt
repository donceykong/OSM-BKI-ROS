cmake_minimum_required(VERSION 3.8)
project(osm_bki)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message("OpenMP enabled")
endif()

find_package(OpenCV REQUIRED)

## Enable OpenMP
add_definitions(-DOPENMP)

## Find ament packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
if(yaml-cpp_FOUND)
  message(STATUS "Found yaml-cpp")
endif()
find_package(PCL REQUIRED)

# Find libosmium for OSM parsing
set(LIBOSMIUM_FOUND FALSE)
set(LIBOSMIUM_INCLUDE_DIRS "")

# Try pkg-config first
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(LIBOSMIUM QUIET libosmium)
  if(LIBOSMIUM_FOUND)
    message(STATUS "Found libosmium via pkg-config")
  endif()
endif()

# If pkg-config didn't find it, try find_path
# libosmium2-dev installs headers in /usr/include/osmium
if(NOT LIBOSMIUM_FOUND)
  # Check if /usr/include/osmium exists (most common location)
  if(EXISTS "/usr/include/osmium/io/any_input.hpp")
    set(LIBOSMIUM_INCLUDE_DIRS "/usr/include")
    set(LIBOSMIUM_FOUND TRUE)
    message(STATUS "Found libosmium headers at ${LIBOSMIUM_INCLUDE_DIRS}/osmium")
  elseif(EXISTS "/usr/local/include/osmium/io/any_input.hpp")
    set(LIBOSMIUM_INCLUDE_DIRS "/usr/local/include")
    set(LIBOSMIUM_FOUND TRUE)
    message(STATUS "Found libosmium headers at ${LIBOSMIUM_INCLUDE_DIRS}/osmium")
  else()
    # Try find_path as fallback
    find_path(LIBOSMIUM_INCLUDE_DIRS 
      NAMES osmium/io/any_input.hpp
      PATHS /usr/include /usr/local/include /opt/local/include
    )
    if(LIBOSMIUM_INCLUDE_DIRS)
      set(LIBOSMIUM_FOUND TRUE)
      message(STATUS "Found libosmium headers at ${LIBOSMIUM_INCLUDE_DIRS}")
    endif()
  endif()
endif()

if(LIBOSMIUM_FOUND)
  include_directories(${LIBOSMIUM_INCLUDE_DIRS})
  if(LIBOSMIUM_LIBRARY_DIRS)
    link_directories(${LIBOSMIUM_LIBRARY_DIRS})
  endif()
  if(LIBOSMIUM_LIBRARIES)
    message(STATUS "Found libosmium libraries: ${LIBOSMIUM_LIBRARIES}")
  else()
    message(STATUS "libosmium is header-only, no libraries needed")
  endif()
else()
  message(FATAL_ERROR "libosmium not found! Please install libosmium-dev (Debian/Ubuntu: sudo apt-get install libosmium2-dev)")
endif()

###########
## Build ##
###########

include_directories(
  include/osm_bki/common
  include/osm_bki/mapping
  include/osm_bki/dataset_devkit
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# Common source files
set(COMMON_SOURCE_FILES src/common/point3f.cpp)

# Files for SEMANTIC BGK source
set(MAPPING_SOURCE_FILES
    src/mapping/bkiblock.cpp
    src/mapping/bkioctomap.cpp
    src/mapping/bkioctree_node.cpp
    src/mapping/bkioctree.cpp)

add_executable(mcd_node
  ${MAPPING_SOURCE_FILES}
  ${COMMON_SOURCE_FILES}
  src/common/osm_visualizer.cpp
  src/mcd_node.cpp
)

add_executable(kitti360_node
  ${MAPPING_SOURCE_FILES}
  ${COMMON_SOURCE_FILES}
  src/common/osm_visualizer.cpp
  src/kitti360_node.cpp
)

add_executable(osm_visualizer_node
  src/common/osm_visualizer.cpp
  src/osm_visualizer_node.cpp
)

ament_target_dependencies(mcd_node
  rclcpp
  geometry_msgs
  visualization_msgs
  std_msgs
  sensor_msgs
  nav_msgs
  pcl_conversions
  tf2_eigen
  tf2_ros
  Eigen3
  ament_index_cpp
  OpenCV
  yaml-cpp
)

target_link_libraries(mcd_node
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
  expat
  bz2
  z
)

ament_target_dependencies(kitti360_node
  rclcpp
  geometry_msgs
  visualization_msgs
  std_msgs
  sensor_msgs
  nav_msgs
  pcl_conversions
  tf2_eigen
  tf2_ros
  Eigen3
  ament_index_cpp
  OpenCV
  yaml-cpp
)

target_link_libraries(kitti360_node
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
  expat
  bz2
  z
)

ament_target_dependencies(osm_visualizer_node
  rclcpp
  geometry_msgs
  visualization_msgs
  std_msgs
  ament_index_cpp
  OpenCV
  yaml-cpp
)

target_link_libraries(osm_visualizer_node
  ${OpenCV_LIBRARIES}
  yaml-cpp
  expat
  bz2
  z
)
if(LIBOSMIUM_LIBRARIES)
  target_link_libraries(osm_visualizer_node
    ${LIBOSMIUM_LIBRARIES}
  )
endif()

#############
## Install ##
#############

install(TARGETS
  mcd_node
  kitti360_node
  osm_visualizer_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# data/ is not installed (kept in source tree; use data_dir param or pkg_src_dir)

install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
