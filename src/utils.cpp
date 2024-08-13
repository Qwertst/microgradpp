#include "utils.hpp"
#include <algorithm>
#include "value.hpp"

namespace nn {
Value MSE_loss(const std::vector<Value> &y, const std::vector<Value> &y_pred) {
    Value loss = make_value(0);
    loss->set_label("mse");
    int n = static_cast<int>(y.size());
    for (int i = 0; i < n; ++i) {
        loss = loss + pow((y[i] - y_pred[i]), 2);
    }
    loss = loss / n;
    return loss;
}

std::vector<Value> flatten(const std::vector<std::vector<Value>> &data) {
    std::vector<Value> res;
    std::for_each(data.begin(), data.end(), [&res](const auto &vec) {
        std::for_each(vec.begin(), vec.end(), [&res](const auto &val) {
            res.push_back(val);
        });
    });
    return res;
}
}  // namespace nn
