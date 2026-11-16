#include "windows_application.hpp"
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        return sandbox::WindowsApplication::run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Sandbox startup/rendering failed: " << error.what() << '\n';
        return 1;
    }
}
