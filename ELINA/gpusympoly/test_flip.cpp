#include <cassert>
#include <iostream>
#include "bitflip_utils.hpp"

int main(){
        int value = 1;
        int bit_all = 8;
        int k_th = 7;
        std:: cout << flip_bit(value, k_th, bit_all) << std::endl;
        std::pair<int,int> res = rangeFlipKBitIntPreserve(value,bit_all,1);
        std::cout << res.first << " " << res.second << std::endl;
        return 0;
}