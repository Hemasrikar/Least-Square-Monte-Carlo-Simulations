cmake_minimum_required(VERSION 3.10)
project(MyProgram)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-Wall -Wextra)

set(SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/lsm_types.cpp
    ${CMAKE_SOURCE_DIR}/src/stochastic_processes.cpp
    ${CMAKE_SOURCE_DIR}/src/payoffs.cpp
    ${CMAKE_SOURCE_DIR}/src/basis_functions.cpp
    ${CMAKE_SOURCE_DIR}/src/ols_regressor.cpp
    ${CMAKE_SOURCE_DIR}/src/lsm_pricer.cpp
    ${CMAKE_SOURCE_DIR}/src/convergence_analyser.cpp
)

add_executable(my_program ${SRC_FILES} ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_include_directories(my_program PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(my_program m)

####################### TESTING STUFF STARTS HERE ###########################################################

enable_testing()
add_subdirectory(tests)
