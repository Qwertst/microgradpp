#include <iostream>
#include <vector>
#include <algorithm>
#include "value.hpp"
#include "mlp.hpp"

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
        res = res + input[it] * weights_[it];
    }
    return nonlin_ ? tanh(res) : res;
}

void Neuron::update(double lr) {
    bias_->update(lr);
    std::for_each(weights_.begin(), weights_.end(), [lr](Value &value) {
        value->update(lr);
    });
}

void Neuron::zero_grad() {
    bias_->zero_grad();
    std::for_each(weights_.begin(), weights_.end(), [](Value &value) {
        value->zero_grad();
    });
}

std::ostream &operator<<(std::ostream &os, const Neuron &neuron) {
    os << "Neuron(in: " << neuron.in_size_ << ")\n";
    os << "Bias : " << neuron.bias_ << ", ";
    for (size_t i = 0; i < neuron.in_size_; ++i) {
        os << "Weight " << i << " : " << neuron.weights_[i] << "\n";
    }
    return os;
}


Layer::Layer(size_t in_size, size_t out_size, bool bias, bool nonlin)
    : in_size_(in_size), out_size_(out_size) {
    neurons_.reserve(out_size_);
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

void Layer::update(double lr) {
    std::for_each(neurons_.begin(), neurons_.end(), [lr](Neuron &neuron) {
        neuron.update(lr);
    });
}

void Layer::zero_grad() {
    std::for_each(neurons_.begin(), neurons_.end(), [](Neuron &neuron) {
        neuron.zero_grad();
    });
}

std::ostream &operator<<(std::ostream &os, const Layer &layer) {
    os << "Layer(in: " << layer.in_size_ << ", out: " << layer.out_size_ << ")\n";
    for (size_t i = 0; i < layer.out_size_; ++i) {
        os << "Neuron " << i << " : " << layer.neurons_[i] << '\n';
    }
    return os;
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

std::vector<std::vector<Value>> MLP::operator()(
    const std::vector<std::vector<Value>> &input
) const {
    std::vector<std::vector<Value>> res;
    res.reserve(input.size());
    for (const auto &vec : input) {
        res.push_back(this->operator()(vec));
    }
    return res;
}

void MLP::update(double lr) {
    std::for_each(layers_.begin(), layers_.end(), [lr](Layer &layer) {
        layer.update(lr);
    });
}

void MLP::zero_grad() {
    std::for_each(layers_.begin(), layers_.end(), [](Layer &layer) {
        layer.zero_grad();
    });
}

std::ostream &operator<<(std::ostream &os, const MLP &mlp) {
    os << "MLP(layers:" << mlp.n_layers_ << ")\n";
    for (size_t i = 0; i < mlp.n_layers_; ++i) {
        os << "Layer " << i << " : " << mlp.layers_[i] << '\n';
    }
    return os;
}

}  // namespace nn
