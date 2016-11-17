//
// Created by jianfei on 16-11-16.
//

#ifndef BIGTOPICMODEL_BUFFERMANAGER_H
#define BIGTOPICMODEL_BUFFERMANAGER_H

#include <vector>
#include <mutex>
#include "glog/logging.h"

template<class T>
class BufferManager {
public:
    std::vector<T> Get() {
        std::lock_guard<std::mutex> lock(m);
        int old_size, new_size;
        old_size = free.size();
        if (!free.empty()) {
            auto ret = std::move(free.back());
            free.pop_back();
            return std::move(ret);
        } else {
            return std::vector<T>();
        }
    }

    void Free(std::vector<T> buffer) {
        std::lock_guard<std::mutex> lock(m);
        free.push_back(std::move(buffer));
    }

private:
    std::vector<std::vector<T>> free;
    std::mutex m;
};


#endif //BIGTOPICMODEL_BUFFERMANAGER_H
