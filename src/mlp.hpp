#pragma once
#include <vector>
#include "value.hpp"

namespace nn {

class Neuron {
public:
    Neuron(size_t in_size, bool bias = true, bool nonlin = true);
    Value operator()(const std::vector<Value> &input) const;

private:
    std::vector<Value> weights_;
    Value bias_;
    size_t in_size_;
    bool nonlin_;
};

class Layer {
public:
    Layer(
        size_t in_size,
        size_t out_size,
        bool bias = true,
        bool nonlin = true
    );
    std::vector<Value> operator()(const std::vector<Value> &input) const;

private:
    std::vector<Neuron> neurons_;
    size_t in_size_;
    size_t out_size_;
};

class MLP {
public:
    MLP(size_t in_size, std::vector<size_t> out_sizes);
    std::vector<Value> operator()(const std::vector<Value> &input) const;

private:
    std::vector<Layer> layers_;
    size_t in_size_;
    std::vector<size_t> out_sizes_;
    size_t n_layers_;
};

}  // namespace nn
