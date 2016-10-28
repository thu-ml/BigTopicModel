#ifndef CM_INCLUDED_H
#define CM_INCLUDED_H

// Reuse DCMSparse to implement a lock-free local storage
// TODO: refactor

#include "dcm.h"

class CMSparse: public DCMSparse {
public:
    CMSparse(const int row_size, const int column_size, const int thread_size) :
        DCMSparse(1, 1, row_size, column_size, PartitionType::row_partition, 1, 
				0, thread_size, LocalMergeStyle::separate, -1) { }
};

#endif
