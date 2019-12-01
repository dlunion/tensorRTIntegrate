
#include "test_common.hpp"
#include <math.h>
#include <algorithm>
#include <tuple>

using namespace std;

void softmax(float* ptr, int count) {

	float total = 0;
	float* p = ptr;
	for (int i = 0; i < count; ++i)
		total += exp(*p++);

	p = ptr;
	for (int i = 0; i < count; ++i, ++p)
		*p = exp(*p) / total;
}

int argmax(float* ptr, int count, float* confidence) {

	auto ind = std::max_element(ptr, ptr + count) - ptr;
	if (confidence) *confidence = ptr[ind];
	return ind;
}

vector<tuple<int, float>> topRank(float* ptr, int count, int ntop) {

	vector<tuple<int, float>> result;
	for (int i = 0; i < count; ++i) 
		result.push_back(make_tuple(i, ptr[i]));
	
	std::sort(result.begin(), result.end(), [](tuple<int, float>& a, tuple<int, float>& b) {
		return get<1>(a) > get<1>(b);
	});

	int n = min(ntop, (int)result.size());
	result.erase(result.begin() + n, result.end());
	return result;
}