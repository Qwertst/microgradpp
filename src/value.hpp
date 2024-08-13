#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace nn {

class Value_handler;
using Value = std::shared_ptr<Value_handler>;
Value make_value(double data, std::vector<Value> childs = {});
Value make_value();
void backward(const Value &value);

class Value_handler {
public:
    Value_handler();
    Value_handler(double data, std::vector<Value> childs);
    double get_data() const;
    double get_grad() const;
    void zero_grad();
    void update(double lr);
    void set_label(std::string label);

    friend void backward(const Value &value);
    friend Value operator+(const Value &lhs, const Value &rhs);
    friend Value operator-(const Value &lhs, const Value &rhs);
    friend Value operator*(const Value &lhs, const Value &rhs);
    friend Value operator/(const Value &lhs, const Value &rhs);

    friend Value operator+(double lhs, const Value &rhs);
    friend Value operator-(double lhs, const Value &rhs);
    friend Value operator*(double lhs, const Value &rhs);
    friend Value operator/(double lhs, const Value &rhs);
    friend Value operator+(const Value &lhs, double rhs);
    friend Value operator-(const Value &lhs, double rhs);
    friend Value operator*(const Value &lhs, double rhs);
    friend Value operator/(const Value &lhs, double rhs);

    friend Value operator-(const Value &arg);

    friend Value pow(const Value &arg, double k);
    friend Value relu(const Value &arg);
    friend Value tanh(const Value &arg);
    friend Value exp(const Value &arg);

    friend std::ostream &operator<<(std::ostream &os, const Value &value);

    friend Value;

private:
    double data_;
    double grad_;
    std::vector<Value> prev_;
    std::function<void()> backward_;
    std::string label_;

    static void top_sort(
        const Value &value,
        std::unordered_set<Value> &visited,
        std::vector<Value> &order
    );
};

}  // namespace nn
