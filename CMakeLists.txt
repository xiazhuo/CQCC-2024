cmake_minimum_required(VERSION 3.1)
project(CCF_QDALS)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/lib)

set(QPANDA_INSTALL_DIR "/usr/local")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${QPANDA_INSTALL_DIR}/lib/cmake")
message(STATIC " ${CMAKE_MODULE_PATH}")

set(EXTRA_LDFLAGS)
if (MSVC)
    list(APPEND EXTRA_LDFLAGS
        -D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING
        -D_CRT_SECURE_NO_WARNINGS
        /bigobj)

    if (NOT USE_MSVC_RUNTIME_LIBRARY_DLL)
        foreach (flag
            CMAKE_C_FLAGS
            CMAKE_C_FLAGS_DEBUG
            CMAKE_C_FLAGS_RELEASE
            CMAKE_C_FLAGS_MINSIZEREL
            CMAKE_C_FLAGS_RELWITHDEBINFO
            CMAKE_CXX_FLAGS
            CMAKE_CXX_FLAGS_DEBUG
            CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL
            CMAKE_CXX_FLAGS_RELWITHDEBINFO)

            if (${flag} MATCHES "/MDd")
                string(REGEX REPLACE "/MDd" "/MT" ${flag} "${${flag}}")
            endif()

            if (${flag} MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${flag} "${${flag}}")
            endif()
            
            if (${flag} MATCHES "/W3")
                string(REGEX REPLACE "/W3" "/W0" ${flag} "${${flag}}")
            endif()
        endforeach()
    endif()
elseif (MINGW)
    list(APPEND EXTRA_LDFLAGS
        -w
        -DGTEST_USE_OWN_TR1_TUPLE=1
        -fpermissive)

    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -Og -ggdb -m64 -Wa, -mbig-obj")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -DNDEBUG")
elseif (UNIX)
    list(APPEND EXTRA_LDFLAGS
        -w
        -DGTEST_USE_OWN_TR1_TUPLE=1
        -fPIC
        -fpermissive)

    if (NOT APPLE)
        list(APPEND LIB_LIST
            dl
            pthread)
    endif ()

    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -DNDEBUG")
endif()
add_definitions(${EXTRA_LDFLAGS})

include_directories(${PROJECT_SOURCE_DIR}
                    ${PROJECT_SOURCE_DIR}/QSolver
                    ${PROJECT_SOURCE_DIR}/QSolver/include/
                    ${PROJECT_SOURCE_DIR}/QSolverApplications
                    ${QPANDA_INSTALL_DIR}/include
                    ${QPANDA_INSTALL_DIR}/include/qpanda2
                    ${QPANDA_INSTALL_DIR}/include/qpanda2/Extensions
                    ${QPANDA_INSTALL_DIR}/include/qpanda2/ThirdParty/)

set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")

set(LIB_LIST)
set(EXTRA_LDFLAGS)


# Curl
find_package(CURL REQUIRED)
if (CURL_FOUND)
    option(USE_CURL "find Curl" ON)
    include_directories(${CURL_INCLUDE_DIRS})
	if(MSVC)
	    #set( CURL_LIBRARY libcurl-static )
    elseif(CMAKE_COMPILER_IS_GNUCXX)
	    set (CURL_LIBRARY curl)
    endif()

    list(APPEND LIB_LIST ${CURL_LIBRARY})
	message("Find CURL_LIBRARY: " ${CURL_LIBRARY})
else (CURL_FOUND)
    option(USE_CURL "not find Curl" OFF)
endif (CURL_FOUND)

# openMP
option(FIND_OPENMP "find openMP" ON)
if (FIND_OPENMP)
    find_package(OpenMP)
endif(FIND_OPENMP)
if(OPENMP_FOUND)
    option(USE_OPENMP "find OpenMP" ON)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
    option(USE_OPENMP "not find OpenMP" OFF)
endif(OPENMP_FOUND)

#include_directories(${PROJECT_SOURCE_DIR}/SU2/externals/cgns)
if(NOT CMAKE_BUILD_TYPE)
   message(STATUS ":  No build type selected, default is Release")
   set(CMAKE_BUILD_TYPE "Debug")
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QPANDA REQUIRED)
if(QPANDA_FOUND)
    INCLUDE_DIRECTORIES(${QPANDA_INCLUDE_DIR})
	message("QPANDA FOUND")
	message(STATUS "QPanda install dir: ${QPANDA_INCLUDE_DIR}.")
else(QPANDA_FOUND)
    message("QPANDA NOT FOUND")
endif(QPANDA_FOUND)

add_definitions(-D HAVE_CGNS)

aux_source_directory(. src)
add_executable(${PROJECT_NAME} ${src})
target_link_libraries(${PROJECT_NAME} 
                      ${QPANDA_LIBRARIES} 
                      ${LIB_LIST})