#ifndef __MPI_HELPERS
#define __MPI_HELPERS

#include <iostream>
#include <vector>

#include <stdarg.h>
#include <mpi.h>

/* Likely/Unlikely macros borrowed from MPICH via ARMCI-MPI */

/* These likely/unlikely macros provide static branch prediction hints to the
 * compiler, if such hints are available.  Simply wrap the relevant expression in
 * the macro, like this:
 *
 * if (unlikely(ptr == NULL)) {
 *     // ... some unlikely code path ...
 * }
 *
 * They should be used sparingly, especially in upper-level code.  It's easy to
 * incorrectly estimate branching likelihood, while the compiler can often do a
 * decent job if left to its own devices.
 *
 * These macros are not namespaced because the namespacing is cumbersome.
 */

/* safety guard for now, add a configure check in the future */
#if ( defined(__GNUC__) && (__GNUC__ >= 3) ) || defined(__IBMC__) || defined(__INTEL_COMPILER) || defined(__clang__)
#  define unlikely(x_) __builtin_expect(!!(x_),0)
#  define likely(x_)   __builtin_expect(!!(x_),1)
#else
#  define unlikely(x_) (x_)
#  define likely(x_)   (x_)
#endif

/* MPI_Count does not exist in MPI-2.  Our implementation
 * does not require it and in any case uses MPI_Aint in
 * place of MPI_Count in many places. */
#if MPI_VERSION < 3
typedef MPI_Aint MPI_Count;
#endif

#ifdef BIGMPI_MAX_INT
static const MPI_Count bigmpi_int_max   = BIGMPI_MAX_INT;
static const MPI_Count bigmpi_count_max = (MPI_Count)BIGMPI_MAX_INT*BIGMPI_MAX_INT;
#else
#include <limits.h>
#include <stdint.h>
static const MPI_Count bigmpi_int_max   = INT_MAX;
/* SIZE_MAX corresponds to size_t, which should be what MPI_Aint is. */
static const MPI_Count bigmpi_count_max = SIZE_MAX;
#endif

/* MPI-3 added const to input arguments, which causes
 * incompatibilities if BigMPI passes in const arguments. */
#if MPI_VERSION >= 3
#define BIGMPI_CONST const
#else
#define BIGMPI_CONST
#endif

/*
 * Synopsis
 *
 * int BigMPI_Type_contiguous(MPI_Aint offset,
 *                            MPI_Count count,
 *                            MPI_Datatype   oldtype,
 *                            MPI_Datatype * newtype)
 *
 *  Input Parameters
 *
 *   offset            byte offset of the start of the contiguous chunk
 *   count             replication count (nonnegative integer)
 *   oldtype           old datatype (handle)
 *
 * Output Parameter
 *
 *   newtype           new datatype (handle)
 *
 * Notes
 *
 *   Following the addition of the offset argument, this function no longer
 *   matches the signature of MPI_Type_contiguous.  This may constitute
 *   breaking user experience for some people.  However, the value of
 *   adding it simplies the primary purpose of this function, which is to
 *   do the heavy lifting _inside_ of BigMPI.  In particular, it allows
 *   us to use MPI_Alltoallw instead of MPI_Neighborhood_alltoallw.
 *
 */
//TODO : for convenience I just inline all functions borrowed from BigMPI...
inline int BigMPI_Type_contiguous(MPI_Aint offset, MPI_Count count, MPI_Datatype oldtype, MPI_Datatype * newtype)
{
    /* The count has to fit into MPI_Aint for BigMPI to work. */
    // BTM rewrite
    //if ((uint64_t)count>(uint64_t)bigmpi_count_max) {
    if ((unsigned long long)count>(unsigned long long)bigmpi_count_max) {
        printf("count (%llu) exceeds bigmpi_count_max (%llu)\n",
               (long long unsigned)count, (long long unsigned)bigmpi_count_max);
        fflush(stdout);
    }

#ifdef BIGMPI_AVOID_TYPE_CREATE_STRUCT
    if (offset==0) {
        /* There is no need for this code path in homogeneous execution,
         * but it is useful to exercise anyways. */
        int a, b;
        int prime = BigMPI_Factorize_count(count, &a, &b);
        if (!prime) {
            MPI_Type_vector(a, b, b, oldtype, newtype);
            return MPI_SUCCESS;
        }
    }
#endif
    MPI_Count c = count/bigmpi_int_max;
    MPI_Count r = count%bigmpi_int_max;

    assert(c<bigmpi_int_max);
    assert(r<bigmpi_int_max);

    MPI_Datatype chunks;
    MPI_Type_vector(c, bigmpi_int_max, bigmpi_int_max, oldtype, &chunks);

    MPI_Datatype remainder;
    MPI_Type_contiguous(r, oldtype, &remainder);

    MPI_Aint lb /* unused */, extent;
    MPI_Type_get_extent(oldtype, &lb, &extent);

    MPI_Aint remdisp          = (MPI_Aint)c*bigmpi_int_max*extent;
    int blocklengths[2]       = {1,1};
    MPI_Aint displacements[2] = {offset,offset+remdisp};
    MPI_Datatype types[2]     = {chunks,remainder};
    MPI_Type_create_struct(2, blocklengths, displacements, types, newtype);

    MPI_Type_free(&chunks);
    MPI_Type_free(&remainder);

    return MPI_SUCCESS;
}

/* Raise an internal fatal BigMPI error.
 *
 * @param[in] file Current file name (__FILE__)
 * @param[in] line Current line numeber (__LINE__)
 * @param[in] func Current function name (__func__)
 * @param[in] msg  Message to be printed
 * @param[in] code Exit error code
 */
inline void BigMPI_Error_impl(const char *file, const int line, const char *func, const char *msg, ...)
{
    va_list ap;
    int  disp;
    char string[500];

    disp  = 0;
    va_start(ap, msg);
    disp += vsnprintf(string, 500, msg, ap);
    va_end(ap);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    fprintf(stderr, "[%d] BigMPI Internal error in %s (%s:%d)\n[%d] Message: %s\n",
            rank, func, file, line, rank, string);
    MPI_Abort(MPI_COMM_WORLD, 100);
}
#define BigMPI_Error(...) BigMPI_Error_impl(__FILE__,__LINE__,__func__,__VA_ARGS__)

/*
 * Synopsis
 *
 * void BigMPI_Convert_vectors(..)
 *
 *  Input Parameter
 *
 *  int          num                length of all vectors (unless splat true)
 *  int          splat_old_count    if non-zero, use oldcount instead of iterating over vector (v-to-w)
 *  MPI_Count    oldcount           single count (ignored if splat_old_count==0)
 *  MPI_Count    oldcounts          vector of counts
 *  int          splat_old_type     if non-zero, use oldtype instead of iterating over vector (v-to-w)
 *  MPI_Datatype oldtype            single type (MPI_DATATYPE_NULL if splat_old_type==0)
 *  MPI_Datatype oldtypes           vector of types (NULL if splat_old_type!=0)
 *  int          zero_new_displs    set the displacement to zero (scatter/gather)
 *  MPI_Aint     olddispls          vector of displacements (NULL if zero_new_displs!=0)
 *
 * Output Parameters
 *
 *  int          newcounts
 *  MPI_Datatype newtypes
 *  MPI_Aint     newdispls
 *
 */
inline void BigMPI_Convert_vectors(int                num,
                            int                splat_old_count,
                            const MPI_Count    oldcount,
                            const MPI_Count    oldcounts[],
                            int                splat_old_type,
                            const MPI_Datatype oldtype,
                            const MPI_Datatype oldtypes[],
                            int                zero_new_displs,
                            const MPI_Aint     olddispls[],
                            int          newcounts[],
                            MPI_Datatype newtypes[],
                            MPI_Aint     newdispls[])
{
    assert(splat_old_count || (oldcounts!=NULL));
    assert(splat_old_type  || (oldtypes!=NULL));
    assert(zero_new_displs || (olddispls!=NULL));

    MPI_Aint lb /* unused */, oldextent;
    if (splat_old_type) {
        MPI_Type_get_extent(oldtype, &lb, &oldextent);
    } else {
        /* !splat_old_type implies ALLTOALLW, which implies no displacement zeroing. */
        assert(!zero_new_displs);
    }

    for (int i=0; i<num; i++) {
        /* counts */
        newcounts[i] = 1;

        /* types */
        BigMPI_Type_contiguous(0, splat_old_count ? oldcount : oldcounts[i],
                               splat_old_type  ? oldtype  : oldtypes[i], &newtypes[i]);
        MPI_Type_commit(&newtypes[i]);

        /* displacements */
        MPI_Aint newextent;
        /* If we are not splatting old type, it implies ALLTOALLW,
         * which does not scale the displacement by the type extent,
         * nor would we ever zero the displacements. */
        if (splat_old_type) {
            MPI_Type_get_extent(newtypes[i], &lb, &newextent);
            newdispls[i] = (zero_new_displs ? 0 : olddispls[i]*oldextent/newextent);
        } else {
            newdispls[i] = olddispls[i];
        }
    }
    return;
}

#if MPI_VERSION >= 3

/*
 * Synopsis
 *
 * int BigMPI_Create_graph_comm(MPI_Comm comm_old, int root, MPI_Comm * graph_comm)
 *
 *  Input Parameter
 *
 *   comm_old           MPI communicator from which to create a graph comm
 *   root               integer id of root.  if -1, create fully connected graph,
 *                      which is appropriate for the all___ collectives.
 *
 * Output Parameters
 *
 *   graph_comm         MPI topology communicator associated with input communicator
 *   rc                 returns the rc from the MPI graph comm create function.
 *
 */
inline int BigMPI_Create_graph_comm(MPI_Comm comm_old, int root, MPI_Comm * comm_dist_graph)
{
    int rank, size;
    MPI_Comm_rank(comm_old, &rank);
    MPI_Comm_size(comm_old, &size);

    /* in the all case (root == -1), every rank is a destination for every other rank;
     * otherwise, only the root is a destination. */
    int indegree  = (root == -1 || root==rank) ? size : 0;
    /* in the all case (root == -1), every rank is a source for every other rank;
     * otherwise, all non-root processes are the source for only one rank (the root). */
    int outdegree = (root == -1 || root==rank) ? size : 1;

    int * sources      = (int *)malloc(indegree*sizeof(int));  assert(sources!=NULL);
    int * destinations = (int *)malloc(outdegree*sizeof(int)); assert(destinations!=NULL);

    for (int i=0; i<indegree; i++) {
        sources[i]      = i;
    }
    for (int i=0; i<outdegree; i++) {
        destinations[i] = (root == -1 || root==rank) ? i : root;
    }

//    BTM rewrite
//    int rc = MPI_Dist_graph_create_adjacent(comm_old,
//                indegree,  sources,      indegree==0  ? MPI_WEIGHTS_EMPTY : MPI_UNWEIGHTED,
//                outdegree, destinations, outdegree==0 ? MPI_WEIGHTS_EMPTY : MPI_UNWEIGHTED,
//                MPI_INFO_NULL, 0 /* reorder */, comm_dist_graph);
    int rc = 0;

    free(sources);
    free(destinations);

    return rc;
}

#endif

typedef enum { GATHERV,
    SCATTERV,
    ALLGATHERV,
    ALLTOALLV,
    ALLTOALLW } bigmpi_collective_t;

typedef enum { ALLTOALLW_OFFSET,
#if MPI_VERSION >= 3
    NEIGHBORHOOD_ALLTOALLW,
#endif
    P2P,
    RMA } bigmpi_method_t;

static pthread_once_t BigMPI_vcollectives_method_is_initialized = PTHREAD_ONCE_INIT;
static bigmpi_method_t BigMPI_vcollectives_method;

/* Tries to deduce collective operation implementation strategy from
   environment */
inline void BigMPI_Detect_default_vcollectives_method()
{
    char *env_var = getenv("BIGMPI_DEFAULT_METHOD");

    if (env_var != NULL) {
#if MPI_VERSION >= 3
        if (strcmp(env_var, "NEIGHBORHOOD_ALLTOALLW")) {
            BigMPI_vcollectives_method = NEIGHBORHOOD_ALLTOALLW;
            return;
        }
#endif

        if (strcmp(env_var, "P2P")) {
            BigMPI_vcollectives_method = P2P;
            return;
        }

        if (strcmp(env_var, "RMA")) {
            BigMPI_vcollectives_method = RMA;
            return;
        }

        fprintf(stderr, "Unknown value \"%s\" for environment variable BIGMPI_DEFAULT_METHOD\n", env_var);
    }

    // fallback to default:
    BigMPI_vcollectives_method = P2P;
}

inline int MPIX_Isend_x(BIGMPI_CONST void *buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)
{
    int rc = MPI_SUCCESS;

    if (likely (count <= bigmpi_int_max )) {
        rc = MPI_Isend(buf, (int)count, datatype, dest, tag, comm, request);
    } else {
        MPI_Datatype newtype;
        BigMPI_Type_contiguous(0,count, datatype, &newtype);
        MPI_Type_commit(&newtype);
        rc = MPI_Isend(buf, 1, newtype, dest, tag, comm, request);
        MPI_Type_free(&newtype);
    }
    return rc;
}

inline int MPIX_Irecv_x(void *buf, MPI_Count count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request * request)
{
    int rc = MPI_SUCCESS;

    if (likely (count <= bigmpi_int_max )) {
        rc = MPI_Irecv(buf, (int)count, datatype, source, tag, comm, request);
    } else {
        MPI_Datatype newtype;
        BigMPI_Type_contiguous(0,count, datatype, &newtype);
        MPI_Type_commit(&newtype);
        rc = MPI_Irecv(buf, 1, newtype, source, tag, comm, request);
        MPI_Type_free(&newtype);
    }
    return rc;
}

inline bigmpi_method_t BigMPI_Get_default_vcollectives_method()
{
    pthread_once(&BigMPI_vcollectives_method_is_initialized, BigMPI_Detect_default_vcollectives_method);
    return BigMPI_vcollectives_method;
}

inline int BigMPI_Collective(bigmpi_collective_t coll, bigmpi_method_t method,
                      BIGMPI_CONST void *sendbuf,
                      const MPI_Count sendcount, const MPI_Count sendcounts[],
                      const MPI_Aint senddispls[],
                      const MPI_Datatype sendtype, const MPI_Datatype sendtypes[],
                      void *recvbuf,
                      const MPI_Count recvcount, const MPI_Count recvcounts[],
                      const MPI_Aint recvdispls[],
                      const MPI_Datatype recvtype, const MPI_Datatype recvtypes[],
                      int root,
                      MPI_Comm comm)
{
    int rc = MPI_SUCCESS;

    int is_intercomm;
    MPI_Comm_test_inter(comm, &is_intercomm);
    if (is_intercomm)
        BigMPI_Error("BigMPI does not support intercommunicators yet.\n");

    if (sendbuf==MPI_IN_PLACE)
        BigMPI_Error("BigMPI does not support in-place in the v-collectives.  Sorry. \n");

    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    if (method==P2P) {

        switch(coll) {
            case ALLTOALLW: /* See page 173 of MPI-3 */
            {
                MPI_Request * reqs = (MPI_Request*)malloc(2*size*sizeof(MPI_Request)); assert(reqs!=NULL);
                /* No extent calculation because alltoallw does not use that. */
                /* Use tag=0 because there is perfect pair-wise matching. */
                for (int i=0; i<size; i++) {
                    /* Pre-post all receives... */
                    MPIX_Irecv_x(recvbuf+recvdispls[i], recvcounts[i], recvtypes[i],
                                 i /* source */, 0 /* tag */, comm, &reqs[i]);
                }
                for (int j=rank; j<(size+rank); j++) {
                    /* Schedule communication in balanced way... */
                    int i = j%size;
                    MPIX_Isend_x(sendbuf+senddispls[i], sendcounts[i], sendtypes[i],
                                 i /* target */, 0 /* tag */, comm, &reqs[size+i]);
                }
                MPI_Waitall(2*size, reqs, MPI_STATUSES_IGNORE);
                free(reqs);
            }
                break;
            case ALLTOALLV: /* See page 171 of MPI-3 */
            {
                MPI_Request * reqs = (MPI_Request*)malloc(2*size*sizeof(MPI_Request)); assert(reqs!=NULL);
                MPI_Aint lb /* unused */, sendextent, recvextent;
                MPI_Type_get_extent(sendtype, &lb, &sendextent);
                MPI_Type_get_extent(recvtype, &lb, &recvextent);
                /* Use tag=0 because there is perfect pair-wise matching without it. */
                for (int i=0; i<size; i++) {
                    /* Pre-post all receives... */
                    MPIX_Irecv_x(recvbuf+recvdispls[i]*recvextent, recvcounts[i], recvtype,
                                 i /* source */, 0 /* tag */, comm, &reqs[i]);
                }
                for (int j=rank; j<(size+rank); j++) {
                    /* Schedule communication in balanced way... */
                    int i = j%size;
                    MPIX_Isend_x(sendbuf+senddispls[i]*sendextent, sendcounts[i], sendtype,
                                 i /* target */, 0 /* tag */, comm, &reqs[size+i]);
                }
                MPI_Waitall(2*size, reqs, MPI_STATUSES_IGNORE);
                free(reqs);
            }
                break;
            case ALLGATHERV:
            {
                MPI_Request * reqs = (MPI_Request*)malloc(2*size*sizeof(MPI_Request)); assert(reqs!=NULL);
                MPI_Aint lb /* unused */, recvextent;
                MPI_Type_get_extent(recvtype, &lb, &recvextent);
                /* Use tag=0 because there is perfect pair-wise matching without it. */
                for (int i=0; i<size; i++) {
                    /* Pre-post all receives... */
                    MPIX_Irecv_x(recvbuf+recvdispls[i]*recvextent, recvcounts[i], recvtype,
                                 i /* source */, 0 /* tag */, comm, &reqs[i]);
                }
                for (int j=rank; j<(size+rank); j++) {
                    /* Schedule communication in balanced way... */
                    int i = j%size;
                    MPIX_Isend_x(sendbuf, sendcount, sendtype,
                                 i /* target */, 0 /* tag */, comm, &reqs[size+i]);
                }
                MPI_Waitall(2*size, reqs, MPI_STATUSES_IGNORE);
                free(reqs);
            }
                break;
            case GATHERV:
            {
                int nreqs = (rank==root ? size+1 : 1);
                MPI_Request * reqs = (MPI_Request*)malloc(nreqs*sizeof(MPI_Request)); assert(reqs!=NULL);
                if (rank==root) {
                    MPI_Aint lb /* unused */, recvextent;
                    MPI_Type_get_extent(recvtype, &lb, &recvextent);
                    for (int i=0; i<size; i++) {
                        /* Use tag=0 because there is perfect pair-wise matching without it. */
                        MPIX_Irecv_x(recvbuf+recvdispls[i]*recvextent, recvcounts[i], recvtype,
                                     i /* source */, 0 /* tag */, comm, &reqs[i+1]);
                    }
                }
                MPIX_Isend_x(sendbuf, sendcount, sendtype,
                             root /* target */, 0 /* tag */, comm, &reqs[0]);
                MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
                free(reqs);
            }
                break;
            case SCATTERV:
            {
                int nreqs = (rank==root ? size+1 : 1);
                MPI_Request * reqs = (MPI_Request*)malloc(nreqs*sizeof(MPI_Request)); assert(reqs!=NULL);
                if (rank==root) {
                    MPI_Aint lb /* unused */, sendextent;
                    MPI_Type_get_extent(sendtype, &lb, &sendextent);
                    for (int i=0; i<size; i++) {
                        /* Use tag=0 because there is perfect pair-wise matching without it. */
                        MPIX_Isend_x(sendbuf+senddispls[i]*sendextent, sendcounts[i], sendtype,
                                     i /* target */, 0 /* tag */, comm, &reqs[i+1]);
                    }
                }
                MPIX_Irecv_x(recvbuf, recvcount, recvtype,
                             root /* source */, 0 /* tag */, comm, &reqs[0]);
                MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
                free(reqs);
            }
                break;
            default:
                BigMPI_Error("Invalid collective chosen. \n");
                break;
        }
#if MPI_VERSION >= 3
    } else if (method==NEIGHBORHOOD_ALLTOALLW) {

        int          * newsendcounts = (int*)malloc(size*sizeof(int));          assert(newsendcounts!=NULL);
        MPI_Datatype * newsendtypes  = (MPI_Datatype*)malloc(size*sizeof(MPI_Datatype)); assert(newsendtypes!=NULL);
        MPI_Aint     * newsdispls    = (MPI_Aint*)malloc(size*sizeof(MPI_Aint));     assert(newsdispls!=NULL);

        int          * newrecvcounts = (int*)malloc(size*sizeof(int));          assert(newrecvcounts!=NULL);
        MPI_Datatype * newrecvtypes  = (MPI_Datatype*)malloc(size*sizeof(MPI_Datatype)); assert(newrecvtypes!=NULL);
        MPI_Aint     * newrdispls    = (MPI_Aint*)malloc(size*sizeof(MPI_Aint));     assert(newrdispls!=NULL);

        switch(coll) {
            case ALLTOALLW:
                assert(root == -1);
                BigMPI_Convert_vectors(size,
                                       0 /* splat count */, 0, sendcounts,
                                       0 /* splat type */, 0, sendtypes,
                                       0 /* zero displs */, senddispls,
                                       newsendcounts, newsendtypes, newsdispls);
                BigMPI_Convert_vectors(size,
                                       0 /* splat count */, 0, recvcounts,
                                       0 /* splat type */, 0, recvtypes,
                                       0 /* zero displs */, recvdispls,
                                       newrecvcounts, newrecvtypes, newrdispls);
                break;
            case ALLTOALLV:
                assert(root == -1);
                BigMPI_Convert_vectors(size,
                                       0 /* splat count */, 0, sendcounts,
                                       1 /* splat type */, sendtype, NULL,
                                       0 /* zero displs */, senddispls,
                                       newsendcounts, newsendtypes, newsdispls);
                BigMPI_Convert_vectors(size,
                                       0 /* splat count */, 0, recvcounts,
                                       1 /* splat type */, recvtype, NULL,
                                       0 /* zero displs */, recvdispls,
                                       newrecvcounts, newrecvtypes, newrdispls);
                break;
            case ALLGATHERV:
                assert(root == -1);
                BigMPI_Convert_vectors(size,
                                       1 /* splat count */, sendcount, NULL,
                                       1 /* splat type */, sendtype, NULL,
                                       1 /* zero displs */, NULL,
                                       newsendcounts, newsendtypes, newsdispls);
                BigMPI_Convert_vectors(size,
                                       0 /* splat count */, 0, recvcounts,
                                       1 /* splat type */, recvtype, NULL,
                                       0 /* zero displs */, recvdispls,
                                       newrecvcounts, newrecvtypes, newrdispls);
                break;
            case GATHERV:
                assert(root != -1);
                BigMPI_Convert_vectors(size,
                                       1 /* splat count */, sendcount, NULL,
                                       1 /* splat type */, sendtype, NULL,
                                       1 /* zero displs */, NULL,
                                       newsendcounts, newsendtypes, newsdispls);
                /* Gatherv: Only the root receives data. */
                if (rank==root) {
                    BigMPI_Convert_vectors(size,
                                           0 /* splat count */, 0, recvcounts,
                                           1 /* splat type */, recvtype, NULL,
                                           0 /* zero displs */, recvdispls,
                                           newrecvcounts, newrecvtypes, newrdispls);
                } else {
                    BigMPI_Convert_vectors(size,
                                           1 /* splat count */, 0, NULL,
                                           1 /* splat type */, MPI_DATATYPE_NULL, NULL,
                                           1 /* zero displs */, NULL,
                                           newrecvcounts, newrecvtypes, newrdispls);
                }
                break;
            case SCATTERV:
                assert(root != -1);
                /* Scatterv: Only the root sends data. */
                if (rank==root) {
                    BigMPI_Convert_vectors(size,
                                           0 /* splat count */, 0, sendcounts,
                                           1 /* splat type */, sendtype, NULL,
                                           0 /* zero displs */, senddispls,
                                           newsendcounts, newsendtypes, newsdispls);
                } else {
                    BigMPI_Convert_vectors(size,
                                           1 /* splat count */, 0, NULL,
                                           1 /* splat type */, MPI_DATATYPE_NULL, NULL,
                                           1 /* zero displs */, NULL,
                                           newsendcounts, newsendtypes, newsdispls);
                }
                BigMPI_Convert_vectors(size,
                                       1 /* splat count */, recvcount, NULL,
                                       1 /* splat type */, recvtype, NULL,
                                       1 /* zero displs */, NULL,
                                       newrecvcounts, newrecvtypes, newrdispls);
                break;
            default:
                BigMPI_Error("Invalid collective chosen. \n");
                break;
        }

        MPI_Comm comm_dist_graph;
        BigMPI_Create_graph_comm(comm, root, &comm_dist_graph);
        rc = MPI_Neighbor_alltoallw(sendbuf, newsendcounts, newsdispls, newsendtypes,
                                    recvbuf, newrecvcounts, newrdispls, newrecvtypes, comm_dist_graph);
        MPI_Comm_free(&comm_dist_graph);

        for (int i=0; i<size; i++) {
            MPI_Type_free(&newsendtypes[i]);
            MPI_Type_free(&newrecvtypes[i]);
        }
        free(newsendcounts);
        free(newsdispls);
        free(newsendtypes);

        free(newrecvcounts);
        free(newrecvtypes);
        free(newrdispls);

#endif
    } else if (method==RMA) {

        /* TODO Add MPI_Win_create_dynamic version of this?
         *      This may entail an MPI_Allgather over base addresses,
         *      but MPI_Win_create has to do that, plus some, so it
         *      may work out to be faster. */

        printf("RMA implementation of v-collectives is incomplete!\n");
        MPI_Abort(comm, 1);

        /* In the RMA implementation, we will treat send as source (buf) and recv as target (win). */
        MPI_Win win;
        /* This is the most (?) conservative approach possible, and assumes that datatypes are
         * noncontiguous and potentially out-of-order. */
        MPI_Aint max_size = 0;
        for (int i=0; i<size; i++) {
            MPI_Aint lb /* unused */, extent;
            MPI_Type_get_extent(recvtypes[i], &lb, &extent);
            MPI_Aint offset = recvdispls[i]+recvcounts[i]*extent;
            max_size = ((offset > max_size) ? offset : max_size);
        }
        MPI_Win_create(recvbuf, max_size, 1, MPI_INFO_NULL, comm, &win);
        MPI_Win_fence(MPI_MODE_NOPRECEDE | MPI_MODE_NOSTORE, win);
        for (int i=0; i<size; i++) {
            MPI_Put(sendbuf+senddispls[i], sendcounts[i], sendtypes[i],
                    i, recvdispls[i], recvcounts[i], recvtypes[i], win);
        }
        MPI_Win_fence(MPI_MODE_NOSUCCEED | MPI_MODE_NOSTORE, win);
        MPI_Win_free(&win);

    } else {
        /* This should be unreachable... */
        BigMPI_Error("Invalid method for v-collectives chosen. \n");
    }
    return rc;
}

inline int MPIX_Allgatherv_x(BIGMPI_CONST void *sendbuf, MPI_Count sendcount, MPI_Datatype sendtype,
                      void *recvbuf, const MPI_Count recvcounts[], const MPI_Aint rdispls[], MPI_Datatype recvtype,
                      MPI_Comm comm)
{
    bigmpi_method_t method = BigMPI_Get_default_vcollectives_method();
    return BigMPI_Collective(ALLGATHERV, method,
                             sendbuf, sendcount, NULL, NULL, sendtype, NULL,
                             recvbuf, -1 /* recvcount */, recvcounts, rdispls, recvtype, NULL,
                             -1 /* root */, comm);
}

inline int MPIX_Alltoallv_x(BIGMPI_CONST void *sendbuf, const MPI_Count sendcounts[], const MPI_Aint sdispls[], MPI_Datatype sendtype,
                     void *recvbuf, const MPI_Count recvcounts[], const MPI_Aint rdispls[], MPI_Datatype recvtype,
                     MPI_Comm comm)
{
    bigmpi_method_t method = BigMPI_Get_default_vcollectives_method();
    return BigMPI_Collective(ALLTOALLV, method,
                             sendbuf, -1 /* sendcount */, sendcounts, sdispls, sendtype, NULL,
                             recvbuf, -1 /* recvcount */, recvcounts, rdispls, recvtype, NULL,
                             -1 /* root */, comm);
}

class MPIHelpers {
public:
    template <class T>
    static void Alltoallv(MPI_Comm comm, int num_nodes,
                          std::vector<size_t> &send_offsets,
                          T *send_buffer,
                          std::vector<size_t> &recv_offsets,
                          std::vector<T> &recv_buffer) {
        // TODO: use BigMPI?
        std::vector<MPI_Count> send_counts(num_nodes);
        std::vector<MPI_Aint> send_displs(num_nodes);
        for (int i=0; i<num_nodes; i++) {
            send_counts[i] = (send_offsets[i+1] - send_offsets[i]) * sizeof(T);
            send_displs[i] = (send_offsets[i]) * sizeof(T);
        }
        std::vector<MPI_Count> recv_counts(num_nodes);
        std::vector<MPI_Aint> recv_displs(num_nodes);
        recv_offsets.resize(num_nodes + 1);
        MPI_Alltoall(send_counts.data(), 1, MPI_LONG_LONG,
                     recv_counts.data(), 1, MPI_LONG_LONG,
                     comm);
        for (int i=1; i<num_nodes; i++) {
            recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
            recv_offsets[i] = recv_displs[i] / sizeof(T);
        }
        size_t sum = recv_offsets.back() = (recv_displs.back() + recv_counts.back()) / sizeof(T);
        recv_buffer.resize(sum);
        MPIX_Alltoallv_x((char*)send_buffer, send_counts.data(), send_displs.data(), MPI_CHAR,
                         (char*)recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_CHAR,
                         comm);
    }

    template <class T>
    static void Allgatherv(MPI_Comm comm, int num_nodes,
                           size_t send_count,
                           T *send_buffer,
                           std::vector<size_t> &recv_offsets,
                           std::vector<T> &recv_buffer) {
        // TODO use BigMPI?
        recv_offsets.resize(num_nodes+1);
        MPI_Allgather(&send_count, 1, MPI_UNSIGNED_LONG_LONG,
                      recv_offsets.data()+1, 1, MPI_UNSIGNED_LONG_LONG, comm);

        std::vector<MPI_Count> recv_counts(num_nodes);
        std::vector<MPI_Aint> recv_displs(num_nodes);
        for (int i=1; i<=num_nodes; i++) {
            recv_counts[i-1] = recv_offsets[i] * sizeof(T);
            recv_offsets[i] += recv_offsets[i-1];
            if (i<num_nodes) recv_displs[i] = recv_offsets[i] * sizeof(T);
        }
        recv_buffer.resize(recv_offsets.back());

        MPIX_Allgatherv_x((char*)send_buffer, send_count * sizeof(T), MPI_CHAR,
                          (char*)recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_CHAR,
                          comm);
    }

    template <class T>
    static void Allgatherv(MPI_Comm comm, int num_nodes,
                           size_t send_count,
                           T *send_buffer,
                           std::vector<size_t> &recv_offsets,
                           T* recv_buffer, size_t buff_size) {
        // TODO use BigMPI?
        recv_offsets.resize(num_nodes+1);
        MPI_Allgather(&send_count, 1, MPI_UNSIGNED_LONG_LONG,
                      recv_offsets.data()+1, 1, MPI_UNSIGNED_LONG_LONG, comm);

        std::vector<MPI_Count> recv_counts(num_nodes);
        std::vector<MPI_Aint> recv_displs(num_nodes);
        for (int i=1; i<=num_nodes; i++) {
            recv_counts[i-1] = recv_offsets[i] * sizeof(T);
            recv_offsets[i] += recv_offsets[i-1];
            if (i<num_nodes) recv_displs[i] = recv_offsets[i] * sizeof(T);
        }
        size_t sum = recv_offsets.back() = (recv_displs.back() + recv_counts.back()) / sizeof(T);
        if (sum > buff_size) {
            std::cout << "Insufficient buffer size " << sum << ' ' << buff_size << std::endl;
        }

        MPIX_Allgatherv_x((char*)send_buffer, send_count * sizeof(T), MPI_CHAR,
                          (char*)recv_buffer, recv_counts.data(), recv_displs.data(), MPI_CHAR,
                          comm);
    }

    // Send `blk_local_first` to worker `proc_id_m1`, and `blk_local_last` to
    // `proc_id_p1`; and receive block of the same size from proc_id_{p,m}1. 
    // Can used to synchronize $\Phi^{t\pm 1}$ in DTM.
    template <typename T>
    static void CircularBlock(const T *blk_local_first, const T *blk_local_last, 
            T *blk_tm1, T *blk_tp1, size_t size_,
            int proc_id_m1, int proc_id_p1, int code_)
    {
        if (proc_id_m1 < 0 && proc_id_p1 < 0) {
            return;
        }
        int size = (int)size_ * sizeof(T);
        auto send = [size](const T *buf_out, T *buf_in, int recv_from, int send_to, int code) {
            MPI_Status status;
            int e;
            if (recv_from == -1) {
                e = MPI_Send(buf_out, size, MPI_CHAR, send_to, code, MPI_COMM_WORLD);
            }
            else if (send_to == -1) {
                e = MPI_Recv(buf_in, size, MPI_CHAR, recv_from, code, MPI_COMM_WORLD, &status);
            }
            else {
                e = MPI_Sendrecv(buf_out, size, MPI_CHAR, send_to, code,
                        buf_in, size, MPI_CHAR, recv_from, code, MPI_COMM_WORLD, &status);
            }
            assert(e == MPI_SUCCESS);
        };
        send(blk_local_last, blk_tm1, proc_id_m1, proc_id_p1, code_);
        send(blk_local_first, blk_tp1, proc_id_p1, proc_id_m1, code_ + 1);
    }
};

#endif
