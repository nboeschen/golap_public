


file(GLOB_RECURSE CORE_CC **.cpp hl/**.cpp **.hpp hl/**.hpp **.h **.cuh)
# Build library
add_library(golap_core INTERFACE ${CORE_CC})
target_link_libraries(golap_core INTERFACE numa cufile stdc++fs ${NVCOMP_LIBRARY})

if (WITH_AWS_CMAKE)
    target_link_libraries(golap_core INTERFACE ${AWSSDK_LINK_LIBRARIES})
endif()
if (WITH_CPU_COMP_CMAKE)
    target_link_libraries(golap_core INTERFACE ${LZ4_LIBRARY} ${SNAPPY_LIBRARY})
endif()

set_target_properties(golap_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(golap_core PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(golap_core PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(golap_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(golap_core PROPERTIES PUBLIC_HEADER CORE_CC)
set_target_properties(golap_core PROPERTIES CUDA_ARCHITECTURES 75)


add_subdirectory(test)
