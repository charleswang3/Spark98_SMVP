# Spark98_SMVP

An optimization of Spark98's shared memory lock-based implementation of sparse matrix vector product (SMVP) using a scheduling approach to partition the sparse matrix array, removing the need for locks and allowing better write-locality at expense of read-locality and pre-processing the matrix.

Authors: George Hatzis-Schoch, Charles Wang
