//
// Created by jianfei on 16-11-15.
//

#include <condition_variable>
#include <mutex>
#include <iostream>
using namespace std;

int main() {
    std::mutex m;
    std::condition_variable cv;
    bool condition = true;

    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [&](){ return condition; });
    cout << "Complete" << endl;
}