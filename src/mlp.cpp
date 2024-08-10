#include "mlp.hpp"
#include "value.hpp"
#include <vector>

namespace nn {

Neuron::Neuron(size_t in_size, bool bias, bool nonlin)
    : in_size_(in_size), nonlin_(nonlin) {
    if (bias) {
        bias_ = make_value();
    }
    weights_.reserve(in_size);
    for (size_t it = 0; it < in_size; ++it) {
        weights_.push_back(make_value());
    }
}

Value Neuron::operator()(const std::vector<Value> &input) const {
    Value res = bias_;
    for (size_t it = 0; it < in_size_; ++it) {
        res = res + input[it]*weights_[it];
    }
    return nonlin_ ? relu(res) : res;
}

Layer::Layer(size_t in_size, size_t out_size, bool bias, bool nonlin)
    : in_size_(in_size), out_size_(out_size) {
    neurons_.resize(out_size_);
    for (size_t it = 0; it < out_size; ++it) {
        neurons_.emplace_back(in_size, bias, nonlin);
    }
}

std::vector<Value> Layer::operator()(const std::vector<Value> &input) const {
    std::vector<Value> res;
    res.reserve(out_size_);
    for (const Neuron &neuron : neurons_) {
        res.push_back(neuron(input));
    }
    return res;
}

MLP::MLP(size_t in_size, std::vector<size_t> out_sizes)
    : in_size_(in_size),
      out_sizes_(std::move(out_sizes)),
      n_layers_(out_sizes_.size()) {
    layers_.emplace_back(in_size, out_sizes_[0]);
    for (size_t it = 1; it < n_layers_; ++it) {
        layers_.emplace_back(
            out_sizes_[it - 1], out_sizes_[it], true, it != n_layers_ - 1
        );
    }
}

std::vector<Value> MLP::operator()(const std::vector<Value> &input) const {
    std::vector<Value> res = input;
    for (const Layer &layer : layers_) {
        res = layer(res);
    }
    return res;
}

}  // namespace nn
