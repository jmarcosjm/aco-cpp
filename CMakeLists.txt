cmake_minimum_required(VERSION 3.20)
project("tcc1" CXX)

# Define as versões mínimas do C++ e do CMake
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Adiciona os diretórios de cabeçalho para a biblioteca dlib e NumCpp
include_directories(/usr/local/include)  # Pasta contendo as pastas de cabeçalho de NumCpp



add_executable(tcc1 main.cpp)

find_package(NumCpp REQUIRED)


target_link_libraries(tcc1
        NumCpp::NumCpp
        )

