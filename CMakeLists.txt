# cmake工程最低版本为3.10
cmake_minimum_required( VERSION 3.10 )

# cmake工程名
project( graphOptimizor )

# 设置编译类型为发布
set( CMAKE_BUILD_TYPE "Release" )

# 添加c++ 17标准的支持
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall" )

# 寻找Eigen库并添加其路径
include_directories( "/usr/include/eigen3" )

# 寻找 TBB 库
find_package( TBB REQUIRED )

# 寻找 ceres 库
find_package( Ceres REQUIRED )

# 添加自定义路径
include_directories( ${PROJECT_SOURCE_DIR} )
include_directories( ${PROJECT_SOURCE_DIR}/include )
include_directories( ${PROJECT_SOURCE_DIR}/src )
include_directories( ${PROJECT_SOURCE_DIR}/test )
include_directories( ${PROJECT_SOURCE_DIR}/test/vertices )
include_directories( ${PROJECT_SOURCE_DIR}/test/edges )

# 添加可执行文件 test_solve_pos
add_executable( test_solve_pos
    test/test_solve_pos.cpp
)
# 添加可执行文件 test_solve_invdep
add_executable( test_solve_invdep
    test/test_solve_invdep.cpp
)
# 添加可执行文件 test_problem
add_executable( test_problem
    test/test_problem.cpp
)
# 添加可执行文件 test_marginalize_pos
add_executable( test_marginalize_pos
    test/test_marginalize_pos.cpp
)
# 添加可执行文件 test_zero_noise
add_executable( test_zero_noise
    test/test_zero_noise.cpp
)
# 添加可执行文件 test_marginalize_invdep_oldest
add_executable( test_marginalize_invdep_oldest
    test/test_marginalize_invdep_oldest.cpp
)
# 添加可执行文件 test_marginalize_invdep_subnew
add_executable( test_marginalize_invdep_subnew
    test/test_marginalize_invdep_subnew.cpp
)
# 添加可执行文件 compare_ceres
add_executable( compare_ceres
    test/compare_with_ceres.cpp
)




# 链接库到可执行文件
target_link_libraries( test_solve_pos
    TBB::tbb
)
target_link_libraries( test_solve_invdep
    TBB::tbb
)
target_link_libraries( test_problem
    TBB::tbb
)
target_link_libraries( test_marginalize_pos
    TBB::tbb
)
target_link_libraries( test_zero_noise
    TBB::tbb
)
target_link_libraries( test_marginalize_invdep_oldest
    TBB::tbb
)
target_link_libraries( test_marginalize_invdep_subnew
    TBB::tbb
)
target_link_libraries( compare_ceres
    TBB::tbb
    ${CERES_LIBRARIES}
)
