add_library(${PROJECT_NAME} SHARED
    array.c
    iterator.c
    broadcasting.c
    ops.c
        ../include/numER/array.h
        ../include/numER/broadcasting.h
)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(${PROJECT_NAME}
    PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
)

if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE m)
endif()
