#install headers
install(FILES nsimd.hh DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/nsimd)
dune_add_test(NAME vctest SOURCES nsimdtest.cc LINK_LIBRARIES dunecommon)

add_subdirectory(test)
