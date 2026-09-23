set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID crpa_band_selection)

add_executable(test_${TEST_ID} test_crpa_band_selection.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics scr_coulomb_lib)

# serial: the selection helper has no MPI dependence
add_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
