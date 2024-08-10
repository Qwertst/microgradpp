#include "value.hpp"
#include <cmath>
#include "doctest.h"

using namespace nn;

#define CHECK_EQ_F(a,b) CHECK(std::abs((a)-(b)) < 1e-9)

TEST_CASE("value_basic") {
    Value a = make_value(5);
    Value b = make_value(0);

    CHECK_EQ(a->get_data(), 5);
    CHECK_EQ(b->get_data(), 0);
    CHECK_EQ(a->get_grad(), 0);
    CHECK_EQ(b->get_grad(), 0);
}

TEST_CASE("value_sum") {
    Value a = make_value(1);
    Value b = make_value(2);

    Value c = a + b;

    CHECK_EQ(c->get_data(), 3);
}

TEST_CASE("value_diff") {
    Value a = make_value(1);
    Value b = make_value(2);

    Value c = a - b;

    CHECK_EQ(c->get_data(), -1);

    Value d = -b;

    CHECK_EQ(d->get_data(), -2);
}

TEST_CASE("value_mul") {
    Value a = make_value(2);
    Value b = make_value(3);

    Value c = a * b;

    CHECK_EQ(c->get_data(), 6);
}

TEST_CASE("value_pow") {
    Value a = make_value(2);

    Value b = pow(a, 5);
    CHECK_EQ(b->get_data(), 32);

    Value c = pow(a, 0);
    CHECK_EQ(c->get_data(), 1);

    Value d = pow(a,-1);
    CHECK_EQ_F(d->get_data(), 0.5);
}

TEST_CASE("value_div") {
    Value a = make_value(50);
    Value b = make_value(5);
    Value c = a / b;

    CHECK_EQ_F(c->get_data(), 10);

    Value x = make_value(1);
    Value y = make_value(3);
    Value z = x / y;

    CHECK_EQ_F(3*z->get_data(), 1);
}

TEST_CASE("value_relu") {
    Value a = make_value(2);

    Value b = relu(a);
    CHECK_EQ(b->get_data(), 2);

    Value aa = make_value(-1);
    Value c = relu(aa);
    CHECK_EQ(c->get_data(), 0);
}

TEST_CASE("value_exp") {
    Value a = make_value(2);

    Value b = exp(a);
    CHECK_EQ(b->get_data(), std::exp(2));

    Value aa = make_value(-1);
    Value c = exp(aa);
    CHECK_EQ(c->get_data(), std::exp(-1));
}

TEST_CASE("value_complex") {
    Value a = make_value(2);

    Value b = pow((-a)*a/a,2);   
    CHECK_EQ(b->get_data(), 4);

    b = relu(-exp(a*a*a));
    CHECK_EQ(b->get_data(), 0);

    b = (exp(make_value(2)*a)-make_value(1)) / (exp(make_value(2)*a)+make_value(1));
    CHECK_EQ_F(b->get_data(), std::tanh(2));
}

TEST_CASE("value_grad_basic") {
    Value a = make_value(5);
    backward(a);

    CHECK_EQ(a->get_grad(), 1);
    
    a->zero_grad();
    CHECK_EQ(a->get_grad(), 0);
}

TEST_CASE("value_grad_sum") {
    Value a = make_value(1);
    Value b = make_value(2);

    Value c = a + b;
    backward(c);

    CHECK_EQ(a->get_grad(), 1);
    CHECK_EQ(b->get_grad(), 1);
}

TEST_CASE("value_grad_sum_itself") {
    Value a = make_value(1);

    Value c = a + a;
    backward(c);

    CHECK_EQ(a->get_grad(), 2);
}

TEST_CASE("value_grad_diff") {
    Value a = make_value(1);
    Value b = make_value(2);

    Value c = a - b;
    backward(c);

    CHECK_EQ(a->get_grad(), 1);
    CHECK_EQ(b->get_grad(), -1);
}

TEST_CASE("value_grad_mul") {
    Value a = make_value(2);
    Value b = make_value(3);

    Value c = a * b;
    backward(c);

    CHECK_EQ(a->get_grad(), 3);
    CHECK_EQ(b->get_grad(), 2);
}

TEST_CASE("value_grad_mul_itself") {
    Value a = make_value(2);

    Value c = a * a * a;
    backward(c);

    CHECK_EQ(a->get_grad(), 3*4);
}

TEST_CASE("value_grad_pow") {
    Value a = make_value(2);

    Value b = pow(a, 5);
    backward(b);
    CHECK_EQ(a->get_grad(), 5*16);

    a->zero_grad();
    b = pow(a, 0);
    backward(b);
    CHECK_EQ(a->get_grad(), 0);

    a->zero_grad();
    b = pow(a,-1);
    backward(b);
    CHECK_EQ_F(a->get_grad(), -0.25);
}

TEST_CASE("value_grad_div") {
    Value a = make_value(4);
    Value b = make_value(2);
    Value c = a / b;

    backward(c);
    CHECK_EQ_F(a->get_grad(), 0.5);
    CHECK_EQ_F(b->get_grad(), -1);
}

TEST_CASE("value_grad_relu") {
    Value a = make_value(2);

    Value b = relu(a);
    backward(b);
    CHECK_EQ(a->get_grad(), 1);

    Value aa = make_value(-1);
    Value c = relu(aa);
    backward(c);
    CHECK_EQ(aa->get_grad(), 0);
}

TEST_CASE("value_grad_exp") {
    Value a = make_value(2);

    Value b = exp(a);
    backward(b);
    CHECK_EQ(a->get_grad(), std::exp(2));
}

TEST_CASE("value_grad_complex") {
    Value a = make_value(2);

    Value s = exp(a)/(make_value(1)+exp(a));
    backward(s);
    CHECK_EQ_F(a->get_grad(), 1/(1+std::exp(-2))*(1-1/(1+std::exp(-2))));

    Value x = make_value(1);
    Value y = make_value(-2);
    Value z = make_value(3);

    Value ss = relu(x+exp(x*y*z+pow(z,2)));
    backward(ss);
    CHECK_EQ_F(x->get_grad(), 1+std::exp(1*(-2)*3+9)*(-2)*3);
    CHECK_EQ_F(y->get_grad(), std::exp(1*(-2)*3+9)*1*3);
    CHECK_EQ_F(z->get_grad(), std::exp(1*(-2)*3+9)*(1*(-2)+2*3));
}