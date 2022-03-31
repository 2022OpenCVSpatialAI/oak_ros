cmake_minimum_required(VERSION 3.5)
project(oak)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_compile_options(-g)

## is used, also find other catkin packages
if(POLICY CMP0057)
    cmake_policy(SET CMP0057 NEW)
endif()

set(_opencv_version 4)
find_package(OpenCV 4 QUIET COMPONENTS imgproc highgui)
if(NOT OpenCV_FOUND)
  message(STATUS "----------------Did not find OpenCV 4, trying OpenCV 3--------------")
  set(_opencv_version 3)
  find_package(OpenCV 3 REQUIRED COMPONENTS imgproc highgui)
endif()
find_package(depthai CONFIG REQUIRED)


find_package(ament_cmake QUIET)
if ( ament_cmake_FOUND )
  message(STATUS "------------------------------------------")
  message(STATUS "Depthai Bridge is being built using AMENT.")
  message(STATUS "------------------------------------------")

  find_package(camera_info_manager REQUIRED)
  find_package(cv_bridge REQUIRED)
  find_package(depthai_ros_msgs REQUIRED)
  find_package(depthai_bridge REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(stereo_msgs REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(vision_msgs REQUIRED)

  set(dependencies
    camera_info_manager
    cv_bridge
    depthai_ros_msgs
    depthai_bridge
    rclcpp
    sensor_msgs
    stereo_msgs
    std_msgs
    vision_msgs
  )

  include_directories(
    include
    ${ament_INCLUDE_DIRS}
  )

  macro(dai_add_node_ros2 node_name node_src)
    add_executable("${node_name}" "${node_src}")

    target_link_libraries("${node_name}"  
                          depthai::core 
                          opencv_imgproc
                          opencv_highgui)

    ament_target_dependencies("${node_name}"  
                              ${dependencies})    

  endmacro() 



          
  ament_package()


elseif( CATKIN_DEVEL_PREFIX OR CATKIN_BUILD_BINARY_PACKAGE )

  find_package(catkin REQUIRED COMPONENTS
    camera_info_manager
    depthai_ros_msgs
    roscpp
    sensor_msgs
    std_msgs
    cv_bridge
    vision_msgs
    depthai_bridge
    message_filters
    nodelet
  )

  catkin_package(
    # INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS depthai_ros_msgs roscpp sensor_msgs std_msgs camera_info_manager depthai_bridge vision_msgs cv_bridge message_filters
    DEPENDS
    OpenCV
  )

  macro(dai_add_node node_name node_src)
      add_executable("${node_name}" "${node_src}")
      
      add_dependencies("${node_name}"
        ${catkin_EXPORTED_TARGETS} 
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
      )

      target_link_libraries("${node_name}" 
        ${catkin_LIBRARIES}
        depthai::core
        opencv_imgproc
        opencv_highgui
      )
      
  endmacro() 

  ## Specify additional locations of header files
  ## Your package locations should be listed before other locations
  include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
  )





endif()
