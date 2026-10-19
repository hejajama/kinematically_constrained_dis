# Vegas from Cuba 4.2.2 (http://www.feynarts.de/cuba/, LGPL, see COPYING),
# reduced to the sources Vegas needs.
#
# Cuba's configure script would write config.h; here the few HAVE_* macros it
# uses are set from CMake checks instead, and HAVE_CONFIG_H is left undefined
# so stddecl.h does not look for config.h. With HAVE_FORK, Cuba evaluates the
# integrand in forked worker processes (number set by CUBACORES or cubacores()).
#
# Everything is compiled with LONGLONGINT, which provides llVegas: 64-bit
# evaluation counts, so that 20 iterations of 1e8 points do not overflow.
# This must apply to Fork.c as well as Vegas.c: they share structures with
# count fields, and with Fork.c built for int counts the forked workers hang
# or return 0. (Cuba's own makefile builds Fork.o once for both widths.)

include(CheckSymbolExists)
include(CheckIncludeFile)
check_symbol_exists(fork "unistd.h" CUBA_HAVE_FORK)
check_symbol_exists(shmget "sys/shm.h" CUBA_HAVE_SHMGET)
check_symbol_exists(getloadavg "stdlib.h" CUBA_HAVE_GETLOADAVG)
check_symbol_exists(erf "math.h" CUBA_HAVE_ERF)
check_include_file(alloca.h CUBA_HAVE_ALLOCA_H)

set(CUBA_DEFINITIONS NOUNDERSCORE=1 LONGLONGINT=1)
foreach (feature FORK SHMGET GETLOADAVG ERF ALLOCA_H)
    if (CUBA_HAVE_${feature})
        list(APPEND CUBA_DEFINITIONS HAVE_${feature}=1)
    endif()
endforeach()
message(STATUS "Cuba definitions: ${CUBA_DEFINITIONS}")

add_library(cuba STATIC
    src/vegas/Vegas.c
    src/common/Fork.c
    src/common/Global.c
    src/common/Data.c
    )
target_include_directories(cuba
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/common ${CMAKE_CURRENT_SOURCE_DIR}/src/vegas)
target_compile_definitions(cuba PRIVATE ${CUBA_DEFINITIONS})
# Cuba does not compile under C23 (it declares bool-like names), so pin C17.
set_target_properties(cuba PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED ON C_EXTENSIONS ON)
# Fork.c creates the worker processes inside an assert(): `assert(socketpair(..)
# != -1 && (pid = fork()) != -1)`. A Release build defines NDEBUG, which would
# compile the fork away and leave Cuba waiting for workers that do not exist.
target_compile_options(cuba PRIVATE -w -UNDEBUG)
find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
    target_link_libraries(cuba PUBLIC ${MATH_LIBRARY})
endif()
