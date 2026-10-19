add_library(dis_lib STATIC
    special_functions.cpp
    rcbk_table.cpp
    interpolation.cpp
    integration.cpp
    )
target_include_directories(dis_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dis_lib PUBLIC cuba)
if (use_boost)
    target_link_libraries(dis_lib PRIVATE Boost::headers)
    target_compile_definitions(dis_lib PRIVATE DIS_BESSEL_BOOST)
else()
    target_link_libraries(dis_lib PRIVATE GSL::gsl GSL::gslcblas)
endif()

add_executable(dis_xs main.cpp)
target_link_libraries(dis_xs PRIVATE dis_lib)
set_target_properties(dis_xs PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
