#include <iostream>
#include "value.hpp"

using namespace nn;

int main() {
    Value a = make_value(1);
    Value e = exp(a)+make_value(1);
    backward(e);
    std::cout << e << a << '\n';
}
