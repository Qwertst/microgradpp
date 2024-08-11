#include "value.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_set>

namespace {
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dist(0, 1);
}  // namespace

namespace nn {

Value make_value(double data, std::vector<Value> childs) {
    return std::make_shared<Value_handler>(data, childs);
}

Value make_value() {
    return std::make_shared<Value_handler>();
}

Value_handler::Value_handler() : data_(dist(gen)), grad_(0) {
}

Value_handler::Value_handler(double data, std::vector<Value> childs)
    : data_(data), grad_(0), prev_(std::move(childs)) {};

double Value_handler::get_data() const {
    return data_;
}

double Value_handler::get_grad() const {
    return grad_;
}

void Value_handler::zero_grad() {
    grad_ = 0;
}

void Value_handler::update(double lr) {
    data_ -= lr * grad_;
}

void Value_handler::set_label(std::string label) {
    label_ = std::move(label);
}

Value operator+(const Value &lhs, const Value &rhs) {
    Value res = make_value(lhs->data_ + rhs->data_, {lhs, rhs});
    Value_handler *weak_res = res.get();
    res->backward_ = [weak_res]() {
        weak_res->prev_[0]->grad_ += weak_res->grad_;
        weak_res->prev_[1]->grad_ += weak_res->grad_;
    };
    return res;
}

Value operator*(const Value &lhs, const Value &rhs) {
    Value res = make_value(lhs->data_ * rhs->data_, {lhs, rhs});
    Value_handler *weak_res = res.get();
    res->backward_ = [weak_res]() {
        weak_res->prev_[0]->grad_ +=
            weak_res->prev_[1]->data_ * weak_res->grad_;
        weak_res->prev_[1]->grad_ +=
            weak_res->prev_[0]->data_ * weak_res->grad_;
    };
    return res;
}

Value pow(const Value &arg, double k) {
    Value res = make_value(std::pow(arg->data_, k), {arg});
    Value_handler *weak_res = res.get();
    res->backward_ = [weak_res, k]() {
        weak_res->prev_[0]->grad_ +=
            k * std::pow(weak_res->prev_[0]->data_, k - 1) * weak_res->grad_;
    };
    return res;
}

Value relu(const Value &arg) {
    Value res = make_value(arg->data_ > 0 ? arg->data_ : 0, {arg});
    Value_handler *weak_res = res.get();
    res->backward_ = [weak_res]() {
        weak_res->prev_[0]->grad_ +=
            (weak_res->data_ > 0 ? 1 : 0) * weak_res->grad_;
    };
    return res;
}

Value tanh(const Value &arg) {
    return (exp(2*arg)-1)/(exp(2*arg)+1);
}

Value exp(const Value &arg) {
    Value res = make_value(std::exp(arg->data_), {arg});
    Value_handler *weak_res = res.get();
    res->backward_ = [weak_res]() {
        weak_res->prev_[0]->grad_ += weak_res->data_ * weak_res->grad_;
    };
    return res;
}

Value operator-(const Value &arg) {
    return arg * make_value(-1);
}

Value operator-(const Value &lhs, const Value &rhs) {
    return lhs + (-rhs);
}

Value operator/(const Value &lhs, const Value &rhs) {
    return lhs * pow(rhs, -1);
}

std::ostream &operator<<(std::ostream &os, const Value &value) {
    os << "(" << value->data_ << " | " << value->grad_;
    if (!value->label_.empty()) {
        os << " | " << value->label_; 
    }
    os << ")";
    return os;
}

void backward(const Value &value) {
    std::unordered_set<Value> visited;
    std::vector<Value> order;

    Value_handler::top_sort(value, visited, order);

    value->grad_ = 1;
    std::for_each(order.rbegin(), order.rend(), [](const Value &node) {
        if (node->backward_) {
            node->backward_();
        }
    });
}

void Value_handler::top_sort(
    const Value &value,
    std::unordered_set<Value> &visited,
    std::vector<Value> &order
) {
    visited.insert(value);
    for (const Value &child : value->prev_) {
        if (visited.find(child) == visited.end()) {
            top_sort(child, visited, order);
        }
    }
    order.push_back(value);
}

Value operator+(double lhs, const Value &rhs) {
    return make_value(lhs) + rhs;
}

Value operator-(double lhs, const Value &rhs) {
    return make_value(lhs) - rhs;
}

Value operator*(double lhs, const Value &rhs) {
    return make_value(lhs) * rhs;
}

Value operator/(double lhs, const Value &rhs) {
    return make_value(lhs) / rhs;
}

Value operator+(const Value &lhs, double rhs) {
    return lhs + make_value(rhs);
}

Value operator-(const Value &lhs, double rhs) {
    return lhs - make_value(rhs);
}

Value operator*(const Value &lhs, double rhs) {
    return lhs * make_value(rhs);
}

Value operator/(const Value &lhs, double rhs) {
    return lhs / make_value(rhs);
}
}  // namespace nn
