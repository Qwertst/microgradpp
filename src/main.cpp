#include "value.hpp"
#include <iostream>

using namespace nn;

int main() {
  Value a = make_value(1);
  Value e = (exp(2 * a) - 1) / (exp(2 * a) + 1);
  backward(e);
  std::cout << e << a << '\n';
}
