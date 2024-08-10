#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>

namespace nn {

class Value_handler;
using Value = std::shared_ptr<Value_handler>;
Value make_value(double data, std::vector<Value> childs = {});
void backward(const Value& value);

class Value_handler {
public:
    Value_handler(double data, std::vector<Value> childs);

    friend void backward(const Value& value);
    friend Value operator+(const Value& lhs, const Value &rhs);
    friend Value operator-(const Value& lhs, const Value &rhs);
    friend Value operator*(const Value& lhs, const Value &rhs);
    friend Value operator/(const Value& lhs, const Value &rhs);
    friend Value operator-(const Value& arg);

    friend Value pow(const Value& arg, double k);
    friend Value relu(const Value& arg);
    friend Value exp(const Value& arg);

    friend std::ostream& operator<<(std::ostream& os, const Value& value);

    friend Value;
private:
    double data_;
    double grad_;
    std::vector<Value> prev_;
    std::function<void()> backward_;

    static void top_sort(const Value& value, std::unordered_set<Value>& visited, std::vector<Value>& order);
};

}
