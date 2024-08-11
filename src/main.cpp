#include <iostream>
#include <vector>
#include "mlp.hpp"
#include "utils.hpp"
#include "value.hpp"

using namespace nn;

int main() {
    std::vector<Value> X;
    std::vector<Value> y;
    int n = 10;
    for (int i = 0; i < n; ++i) {
        double x = i;
        X.push_back(make_value(x));
        y.push_back(make_value(2*x+1));
    }

    Value a = make_value(2);
    Value b = make_value(0);
    a->set_label("a");
    b->set_label("b");

    double lr = 0.01;

    for (int k = 0; k < 10; ++k) {
        std::vector<Value> y_pred;
        y_pred.reserve(n);
        for (int i = 0; i < n; ++i) {
            y_pred.push_back(b+a*X[i]);
        }
        a->zero_grad();
        b->zero_grad();
        Value loss = MSE_loss(y, y_pred);
        backward(loss);
        std::cout << loss << ' ' << b << ' ' << a << '\n';
        a->update(lr);
        b->update(lr);
    }
}
