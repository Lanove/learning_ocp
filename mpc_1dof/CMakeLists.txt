# Authors: Giulio Romualdi
# CopyPolicy: Released under the terms of the LGPLv2.1 or later

cmake_minimum_required(VERSION 3.1)

set (CMAKE_CXX_STANDARD 11)

project(mpc_1dof)

find_package(OsqpEigen)
find_package(Eigen3)

include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

#MPCExample
# add_executable(mpc_1dof mpc.cpp)
# target_link_libraries(mpc_1dof OsqpEigen::OsqpEigen)

add_executable(mpc_1dof_oop mpc_oop.cpp)
target_link_libraries(mpc_1dof_oop OsqpEigen::OsqpEigen)