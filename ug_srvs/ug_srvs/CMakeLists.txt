cmake_minimum_required(VERSION 2.8.3)
project(ug_srvs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  sensor_msgs
  std_msgs
  std_srvs
  ug_msgs
)

################################################
## Declare ROS messages, services and actions ##
################################################



# Generate services in the 'srv' folder
add_service_files(
  SaveImages.srv
  ImagesSequence.srv
  ID.srv
  Onoff.srv
  PoseStamped.srv

)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
    geometry_msgs
    sensor_msgs
    std_msgs
    ug_msgs
)


###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    message_runtime
    sensor_msgs
    std_msgs
    ug_msgs
)