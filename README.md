# Spark98_SMVP

We will optimize Spark98's shared memory lock-based implementation of sparse matrix vector product (SMVP) by implementing a scheduling approach to partitioning the sparse matrix array, removing the need for locks and allowing better write-locality at expense of read-locality and pre-processing the matrix.
