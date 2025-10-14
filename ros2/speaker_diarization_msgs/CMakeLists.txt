cmake_minimum_required(VERSION 3.8)
project(speaker_diarization_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 메시지 파일들 정의
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SpeakerInfo.msg"
)
ament_export_dependencies(rosidl_default_runtime)


ament_package()

