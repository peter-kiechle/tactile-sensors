#include "controller.h"

#include <boost/version.hpp>

int main(int argc, char **argv) {
    Controller controller = Controller(argc, argv); // Model View Controller pattern
    return EXIT_SUCCESS;
}
