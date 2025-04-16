
file(GLOB imagenetSources *.cpp)
file(GLOB imagenetIncludes *.h )

cuda_add_executable(imagenet-stock ${imagenetSources})
target_link_libraries(imagenet-stock jetson-inference)
install(TARGETS imagenet-stock DESTINATION bin)
