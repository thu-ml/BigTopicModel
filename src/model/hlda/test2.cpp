//
// Created by jianfei on 16-11-15.
//

#include <condition_variable>
#include <mutex>
#include <vector>
#include <iostream>
using namespace std;

int main() {
    /*std::mutex m;
    std::condition_variable cv;
    bool condition = true;

    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [&](){ return condition; });
    cout << "Complete" << endl;*/

    std::vector<int> a(3), b;
    b = std::move(a);

    cout << a.size() << ' ' << b.size() << endl;
}