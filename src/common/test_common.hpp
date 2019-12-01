
#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>

std::vector<std::tuple<int, float>> topRank(float* ptr, int count, int ntop = 5);
int argmax(float* ptr, int count, float* confidence = nullptr);
void softmax(float* ptr, int count);


#endif //COMMON_HPP