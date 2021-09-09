#install headers
install(FILES vectorclass.hh DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/vectorclass)
dune_add_test(NAME vctest SOURCES vectorclasstest.cc LINK_LIBRARIES dunecommon)

add_subdirectory(test)
