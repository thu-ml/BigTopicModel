//
// Created by jianfei on 16-11-11.
//

#ifndef BIGTOPICMODEL_CHANNEL_H
#define BIGTOPICMODEL_CHANNEL_H

#include <list>
#include <thread>
#include <mutex>
#include <condition_variable>

template<class item>
class channel {
private:
    std::list<item> queue;
    std::mutex m;
    std::condition_variable cv, cv2;
    bool closed;
public:
    channel() : closed(false) { }
    void close() {
        std::unique_lock<std::mutex> lock(m);
        closed = true;
        cv.notify_all();
    }
    void open() {
        std::unique_lock<std::mutex> lock(m);
        closed = false;
        cv.notify_all();
    }
    bool is_closed() {
        std::unique_lock<std::mutex> lock(m);
        return closed;
    }
    void put(item &i) {
        std::unique_lock<std::mutex> lock(m);
        if(closed)
            throw std::logic_error("put to closed channel");
        queue.push_back(std::move(i));
        cv.notify_one();
    }
    bool get(item &out, bool wait = true) {
        std::unique_lock<std::mutex> lock(m);
        if(wait)
            cv.wait(lock, [&](){ return closed || !queue.empty(); });
        if(queue.empty())
            return false;
        out = std::move(queue.front());
        queue.pop_front();
        cv2.notify_all();
        return true;
    }
    void wait_empty() {
        std::unique_lock<std::mutex> lock(m);
        cv2.wait(lock, [&](){ return queue.empty(); });
    }
};

#endif //BIGTOPICMODEL_CHANNEL_H
