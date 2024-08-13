#pragma once
#include "value.hpp"

namespace nn {
Value MSE_loss(const std::vector<Value> &y, const std::vector<Value> &y_pred);
std::vector<Value> flatten(const std::vector<std::vector<Value>> &data);
}  // namespace nn
