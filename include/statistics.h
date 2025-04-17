#include <cmath> 
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>

double mean(const std::vector<double>& v){
	if(v.empty()){return 0.0;}
	double s = std::accumulate(v.begin(), v.end(), 0.0);
	return s/v.size();
}

double stdev(const std::vector<double>& v){
	if(v.empty()){return 0.0;}
	double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
	double sQ = 0.0;
	for(double valor : v){sQ += std::pow(valor-m, 2);}
	return std::sqrt(sQ/v.size());
}
