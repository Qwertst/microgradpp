#include <iostream>
#include <vector>
#include "mlp.hpp"
#include "utils.hpp"
#include "value.hpp"

using namespace nn;

int main() {
    std::vector<std::vector<Value>> X = {
        {make_value(2.0), make_value(3.0), make_value(-1.0)},
        {make_value(3.0), make_value(-1.0), make_value(0.5)},
        {make_value(0.5), make_value(1.0), make_value(1.0)},
        {make_value(1.0), make_value(1.0), make_value(-1.0)}
    };
    std::vector<Value> y = {
        make_value(1), make_value(-1), make_value(-1), make_value(1)
    };
    double lr = 0.1;

    MLP nnn(3, {4, 4, 1});

    for (int k = 0; k < 20; ++k) {
        auto y_pred = flatten(nnn(X));
        nnn.zero_grad();
        Value loss = MSE_loss(y, y_pred);
        backward(loss);
        std::cout << k << " " << loss << '\n';
        nnn.update(lr);
    }
}
