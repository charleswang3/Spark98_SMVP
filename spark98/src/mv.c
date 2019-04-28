/***************************************************************************/
/* mv.c - sequential and parallel shared memory SMVP kernel                */
/*                                                                         */
/* Spark98 Kernels                                                         */
/* Copyright (C) David O'Hallaron 1998                                     */
/*                                                                         */
/* Compilation options: (you must define zero or one of these)             */
/*                                                                         */
/* Variable       Binary   Description                                     */
/* none           smv      baseline sequential SMVP                        */
/* PTHREAD_LOCK   lmv      locks/pthreads                                  */
/* PTHREAD_REDUCE rmv      reductions instead of locks/pthreads            */
/* SGI_LOCK       lmv      locks/SGI threads                               */
/* SGI_REDUCE     rmv      reductions instead of locks/SGI threads         */
/*                                                                         */
/* You are free to use this software without restriction. If you find that */
/* the suite is helpful to you, it would be very helpful if you sent me a  */
/* note at droh@cs.cmu.edu letting me know how you are using it.           */
/***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#if (defined(PTHREAD_LOCK) || defined(PTHREAD_REDUCE) || defined(SCHEDULE))
#define PTHREAD
#elif (defined(SGI_LOCK) || defined(SGI_REDUCE))
#define SGI
#endif

#if defined(PTHREAD)
#include <pthread.h>
#elif defined(SGI)
#include <sys/types.h>
#include <sys/prctl.h>
#include <ulocks.h>
#endif

#if (defined(SGI_LOCK) || defined(PTHREAD_LOCK))
#define LOCK
#else
#undef LOCK
#endif

#if (defined(SGI_REDUCE) || defined(PTHREAD_REDUCE))
#define REDUCE
#else
#undef REDUCE
#endif

/*
 * program wide constants
 */
#define DEFAULT_ITERS 20   /* default number of SMVP operations */
#define STRLEN 128         /* default string length */
#define DOF 3              /* degrees of freedom in underlying simulation */

struct gi {
  char *progname;              /* program name */

  /* command line options */
  int quiet;                 /* run quietly unless there are errors (-Q) */
  int iters;                 /* number of times to perform SMVP (-i<n>) */
  int output;                /* print the result vector? (-o) */
  int threads;               /* number of threads (-t<n>) */
  int locks;                 /* number of locks (-l<n>) */
  char packfilename[STRLEN]; /* input packfile name */

  /* problem parameters */
  int globalnodes;           /* global nodes (same as local nodes) */
  int globalelems;           /* global elements (same as local elements) */
  int mesh_dim;              /* mesh dimension (3) */
  int corners;               /* nodes per element (4) */
  int subdomains;            /* number of partition sets */
  int processors;            /* not used */
  int nodes;                 /* number of local nodes */
  int elems;                 /* number of local elements */
  int matrixlen;             /* number of local nonzero block entries */
  int choleskylen;           /* not used */
  int priv;                  /* not used */
  int mine;                  /* not used */

  /* global data structures */
  double (*coord)[DOF];      /* geometric node coordinates */
  int (*vertex)[4];          /* nodes associated with each element */
  int *globalnode;           /* global to local node mapping function */
  int *matrixcol;            /* K[i] is in column matrixcol[i] */
  int *matrixindex;          /* row i starts at K[matrixindex[i]] */
};

/*
 * globals
 */

/* w1 = K1 * x1 */
double (*K1)[DOF][DOF];     /* sparse matrix  coefficients */
double (*v1)[DOF];          /* dense vector */
double (*w1)[DOF];          /* dense vector */

/* w2 = K2 * x2 */
double (*K2)[DOF][DOF];     /* sparse matrix  coefficients */
double (*v2)[DOF];          /* dense vector */
double (*w2)[DOF];          /* dense vector */

#if defined(REDUCE)
/* temporary vectors (one per thread) to hold */
/* partially assembled  w vectors */
double (**tmpw)[DOF];
#endif

/* global information about the simulation */
struct gi gi, *gip = &gi;

/* timing variables */
double secs, csecs;

/* partition */
int *firstrow;

/* thread ids */
int *ids;

/*
 * Pthread-specific thread variables
 */
#if defined(PTHREAD)
pthread_mutex_t barriermutexhandle;
pthread_mutex_t *vectormutexhandle;
pthread_cond_t barriercondhandle;
int barriercnt = 0;

/*
 * SGI-specific thread variables
 */
#elif defined(SGI)
char arenaname[STRLEN]= "/dev/zero";
usptr_t *arenahandle;
barrier_t *barrierhandle;
ulock_t *lockhandle; /* array of locks */
#endif

/*
 * end globals
 */

/*
 * function prototypes
 */

/* code executed by each thread */
void *smvpthread(void *a);

/* initialization routines */
void init(int argc, char **argv, struct gi *gip);
void parsecommandline(int argc, char **argv, struct gi *gip);
void readpackfile(struct gi *gip);

/* routines that assemble the sparse matrix */
void assemble_matrix(double (*K)[DOF][DOF], struct gi *gip);
void assemble_vector(double (*v)[DOF], struct gi *gip);

/* sparse matrix vector multiply */
void zero_vector(double (*v)[DOF], int firstrow, int numrows);
void local_smvp(int nodes, double (*A)[DOF][DOF], int *Acol,
		int *Aindex, double (*v)[DOF], double (*w)[DOF],
		int firstrow, int numrows, int id, struct gi *gip);
void add_vectors(double (**tmpw)[DOF], double (*w)[DOF],
		 int firstrow, int numrows, struct gi *gip);

/* misc routines */
void info(struct gi *gip);
void usage(struct gi *gip);
void printfvec3(double (*v)[DOF], int n);
void printmatrix3(double (*A)[DOF][DOF], struct gi *gip);
void printnodevector(double (*v)[DOF], int n);

/* system-dependent timer routines */
void init_etime(void);
double get_etime(void);

/* system dependent thread routines */
#if (defined(LOCK) || defined(REDUCE) || defined(SCHEDULE))
void spark_init_threads(void);
void spark_start_threads(int n);
void spark_barrier(void);
void spark_setlock(int lockid);
void spark_unsetlock(int lockid);
#endif
#if defined(SCHEDULE)
    int **schedule;
    int **schedule_row;
    int *schedule_len;
#endif

/*
 * main program
 */

int find_partition_ind(int i)
{
    int j;
    for (j = 0; j < gip->threads - 1; j++)
    {
        if (i >= gip->matrixindex[firstrow[j]] && i < gip->matrixindex[firstrow[j + 1]])
            return j;
    }
    return gip->threads - 1;
}
void main(int argc, char **argv) {

  int i;
  int oldrow, newrow, limit, nonzeros, minnonzeros, maxnonzeros;
  double mflops;

  init(argc, argv, gip);

  /*
   * allocate contiguous storage for the matrix and vectors
   */
  if (!(K1 = (void *)malloc((gip->matrixlen+1)*DOF*DOF*sizeof(double)))) {
    fprintf(stderr, "%s: couldn't malloc K1(%d)\n", gip->progname,
	    gip->matrixlen);
    exit(0);
  }
  if (!(K2 = (void *)malloc((gip->matrixlen+1)*DOF*DOF*sizeof(double)))) {
    fprintf(stderr, "%s: couldn't malloc K2(%d)\n", gip->progname,
	    gip->matrixlen);
    exit(0);
  }
  if (!(v1 = (void *)malloc(gip->nodes * DOF * sizeof(double)))) {
    fprintf(stderr, "%s: couldn't malloc v1(%d)\n", gip->progname, gip->nodes);
    exit(0);
  }
  if (!(v2 = (void *)malloc(gip->nodes * DOF * sizeof(double)))) {
    fprintf(stderr, "%s: couldn't malloc v2(%d)\n", gip->progname, gip->nodes);
    exit(0);
  }
  if (!(w1 = (void *)malloc(gip->nodes * DOF * sizeof(double)))) {
    fprintf(stderr, "%s: couldn't malloc w1(%d)\n", gip->progname, gip->nodes);
    exit(0);
  }
  if (!(w2 = (void *)malloc(gip->nodes * DOF * sizeof(double)))) {
    fprintf(stderr, "%s: couldn't malloc w2(%d)\n", gip->progname, gip->nodes);
    exit(0);
  }

#if defined(REDUCE)
  if (!(tmpw = (void *)malloc(gip->threads*sizeof(double *)))) {
    fprintf(stderr, "%s: couldn't malloc tmpw(%d)\n",
	    gip->progname, gip->threads);
    exit(0);
  }
  for (i=0; i<gip->threads; i++) {
    if (!(tmpw[i] = (void *)malloc(gip->nodes*DOF*sizeof(double)))) {
      fprintf(stderr, "%s: couldn't malloc tmpw[i](%d)\n",
	      gip->progname, gip->nodes);
      exit(0);
    }
  }
#endif

  /*
   * generate the sparse matrix and vector coefficients
   */
  if (!gip->quiet)
    fprintf(stderr, "%s: Computing sparse matrix coefficients.", argv[0]);
  assemble_matrix(K1, gip);
  assemble_matrix(K2, gip);
  assemble_vector(v1, gip);
  assemble_vector(v2, gip);

  if (!gip->quiet) {
    fprintf(stderr, " Done.\n");
    fflush(stderr);
  }

  /*
   * partition the load across the threads
   */
  limit = gip->matrixlen / gip->threads;
  oldrow = 0;
  newrow = 0;
  firstrow[0] = 0;

  for (i=1; i<gip->threads; i++) {
    if (newrow > gip->nodes) {
      fprintf(stderr, "%s: inconsistent row number\n");
      exit(0);
    }
    else if (newrow < gip->nodes) {
      newrow = oldrow + 1;
      while (((gip->matrixindex[newrow]-gip->matrixindex[oldrow]) < limit)
	     && (newrow < gip->nodes)) {
	newrow++;
      }
      firstrow[i] = newrow;
      oldrow = newrow;
    }
    else {
      firstrow[i] = newrow;
    }
  }
  firstrow[gip->threads] = gip->nodes;
    
  //Allocate and determine schedules
#if defined(SCHEDULE)

    //printf("BEFORE SCHEDULE PROCESSING \n");
    schedule = calloc(gip->threads, sizeof(int**));
    schedule_row = calloc(gip->threads, sizeof(int**));
    schedule_len = calloc(gip->threads, sizeof(int));
    for (i = 0; i < gip->threads; i++)
    {
        schedule[i] = calloc(gip->matrixlen, sizeof(int*));
        schedule_row[i] = calloc(gip->matrixlen, sizeof(int*));
    }
    int r;
    //Fill in schedule
    for (r = 0; r < 30169; r++)
    {
        int rowstart = gip->matrixindex[r];
        for (i = rowstart; i < gip->matrixindex[r + 1]; i++)
        {
            int elem = i;
            int j = gip->matrixcol[i];
            int p1 = find_partition_ind(i);
            schedule[p1][schedule_len[p1]] = elem;
            schedule_row[p1][schedule_len[p1]] = r;
            schedule_len[p1]++;
            int p2 = find_partition_ind(j);
            schedule[p2][schedule_len[p2]] = elem;
            schedule_row[p2][schedule_len[p2]] = j;
            schedule_len[p2]++;
        }
    }
    //printf("After SChedule PROCESSING\n");

#endif

    /*
     * start the workers
     */
#if (defined(LOCK) || defined(REDUCE) || defined(SCHEDULE))
  spark_start_threads(gip->threads-1);
#endif

  /*
   * start the master
   */
  if (!gip->quiet) {
#if defined(LOCK)
      fprintf(stderr, "%s: Performing %d SMVP pairs (n=%d) with %d threads and %d locks.",
	      gip->progname, gip->iters, gip->nodes, gip->threads, gip->locks);
#elif (defined(REDUCE) || defined(SCHEDULE))
      fprintf(stderr, "%s: Performing %d SMVP pairs (n=%d) with %d threads.",
	      gip->progname, gip->iters, gip->nodes, gip->threads);
#else
      fprintf(stderr, "%s: Performing %d SMVP pairs (n=%d).",
	      gip->progname, gip->iters, gip->nodes);
#endif
  }
  ids[0] = 0;
  smvpthread(&ids[0]);

  if (!gip->quiet) {
    fprintf(stderr, " Done.\n");
    fflush(stderr);
  }

  if (secs == 0.0) {
    fprintf(stderr, "error: no measured elapsed time. Use more iterations (e.g., -i%d)\n",
	    gip->iters*10);
    exit(0);
  }

  /*
   * summarize performance
   */
  mflops =
    (double)((2.0*gip->matrixlen - gip->nodes) *   /* nonzero blocks */
     DOF*DOF *                                     /* DOF^2 numbers/block */
     2.0)                                          /* 2 flops/block */
     / 1000000.0;

  /* compute min and max load on each thread */
  minnonzeros = 1<<20;
  maxnonzeros = -1;
  for (i=0; i<gip->threads; i++) {
    nonzeros = gip->matrixindex[firstrow[i+1]] - gip->matrixindex[firstrow[i]];
    if (nonzeros < minnonzeros)
      minnonzeros = nonzeros;
    if (nonzeros > maxnonzeros)
      maxnonzeros = nonzeros;
  }

#if defined(LOCK)
  /* lock-based SMVP */
  fprintf(stderr, "%s: %s %d threads %d locks %.6f Mf %.6f s %.1f Mf/s [%d/%d/%.0f%%]\n",
	  gip->progname, gip->packfilename, gip->threads, gip->locks, mflops, secs, mflops/secs,
	  minnonzeros, maxnonzeros,
	  (double)((double)minnonzeros/(double)maxnonzeros)*100.0);

#elif defined (REDUCE)
  /* reduction-based SMVP */
  fprintf(stderr,
	  "%s: %s %d threads %.6f Mf %.6f s [%.6f/%.6f/%.0f%%] %.1f Mf/s [%d/%d/%.0f%%]\n",
	  gip->progname, gip->packfilename, gip->threads, mflops, secs, secs-csecs, csecs,
	  ((secs-csecs)/secs)*100.0, mflops/secs, minnonzeros, maxnonzeros,
	  (double)((double)minnonzeros/(double)maxnonzeros)*100.0);

#elif defined(SCHEDULE)
fprintf(stderr,
	  "%s: %s %d threads %.6f Mf %.6f s %.1f Mf/s [%d/%d/%.0f%%]\n",
	  gip->progname, gip->packfilename, gip->threads, mflops, secs, mflops/secs, minnonzeros, maxnonzeros,
	  (double)((double)minnonzeros/(double)maxnonzeros)*100.0);

#else
  /* baseline sequential SMVP */
  fprintf(stderr, "%s: %s %.6f Mf %.6f s %.1f Mf/s\n",
	  gip->progname, gip->packfilename, mflops, secs, mflops/secs);
#endif

  /* print results */
  if (gip->output) {
    printnodevector(w1, gip->nodes);
  }
  fflush(stdout);

  if (!gip->quiet) {
    fprintf(stderr, "%s: Done.\n", gip->progname);
  }

  exit(0);
}

/*
 * init - initialize the kernel
 */
void init(int argc, char **argv, struct gi *gip) {

  init_etime();

  gip->progname = argv[0];
  parsecommandline(argc, argv, gip);

  if (!gip->quiet)
    fprintf(stderr, "%s: Reading %s.\n", argv[0], gip->packfilename);
  readpackfile(gip);

  if (gip->locks == 0)
    gip->locks = gip->nodes;

  if (!(firstrow = (int *)malloc((gip->threads+1) * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't malloc firstrow\n", gip->progname);
    exit(0);
  }

  if (!(ids = (int *)malloc((gip->threads) * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't malloc ids\n", gip->progname);
    exit(0);
  }

#if (defined(LOCK) || defined(REDUCE) || defined(SCHEDULE))
  spark_init_threads();
#endif

}

/*
 * smvpthread - perform a sequence of SMVP pairs
 */
void *smvpthread(void *a) {
  int i;
  int *argp = (int *)a;
  int id = *argp;
  double mycsecs, mystarttime, mystartctime;

#if (defined(LOCK) || defined(REDUCE) || defined(SCHEDULE))
  int numrows = firstrow[id+1] - firstrow[id];
  spark_barrier();
#endif

  mycsecs = 0.0;
  mystarttime = get_etime();

  for (i=0; i<gip->iters; i++) {

#if defined(LOCK) /* parallel with one or more locks */

    /* w1 = K1*v1 */
    spark_barrier();
    zero_vector(w1, firstrow[id], numrows);
    spark_barrier();
    local_smvp(gip->nodes, K1, gip->matrixcol, gip->matrixindex,
	       v1, w1, firstrow[id], numrows, id, gip);

    /* w2 = K2*v2 */
    spark_barrier();
    zero_vector(w2, firstrow[id], numrows);
    spark_barrier();
    local_smvp(gip->nodes, K2, gip->matrixcol, gip->matrixindex,
	       v2, w2, firstrow[id], numrows, id, gip);

#elif defined(REDUCE) /* parallel with reductions instead of locks */

    /* w1 = K1*v1 */
    spark_barrier();
    zero_vector(tmpw[id], 0, gip->nodes);
    local_smvp(gip->nodes, K1, gip->matrixcol, gip->matrixindex,
	       v1, tmpw[id], firstrow[id], numrows, id, gip);

    mystartctime = get_etime();
    spark_barrier();
    add_vectors(tmpw, w1, firstrow[id], numrows, gip);
    mycsecs += (get_etime() - mystartctime);

    /* w2 = K2*v2 */
    spark_barrier();
    zero_vector(tmpw[id], 0, gip->nodes);
    local_smvp(gip->nodes, K2, gip->matrixcol, gip->matrixindex,
	       v2, tmpw[id], firstrow[id], numrows, id, gip);

    mystartctime = get_etime();
    spark_barrier();
    add_vectors(tmpw, w2, firstrow[id], numrows, gip);
    mycsecs += (get_etime() - mystartctime);
#elif defined(SCHEDULE)
    /* w1 = K1*v1 */
    spark_barrier();
    zero_vector(w1, firstrow[id], numrows);
    spark_barrier();
    local_smvp(gip->nodes, K1, gip->matrixcol, gip->matrixindex,
	       v1, w1, firstrow[id], numrows, id, gip);

    /* w2 = K2*v2 */
    spark_barrier();
    zero_vector(w2, firstrow[id], numrows);
    spark_barrier();
    local_smvp(gip->nodes, K2, gip->matrixcol, gip->matrixindex,
	       v2, w2, firstrow[id], numrows, id, gip);


#else /* sequential */

    /* w1 = K1*v1 */
    zero_vector(w1, 0, gip->nodes);
    local_smvp(gip->nodes, K1, gip->matrixcol, gip->matrixindex,
	       v1, w1, 0, gip->nodes, id, gip);

    /* w2 = K2*v2 */
    zero_vector(w2, 0, gip->nodes);
    local_smvp(gip->nodes, K2, gip->matrixcol, gip->matrixindex,
	       v2, w2, 0, gip->nodes, id, gip);
#endif

  }

  if (id == 0) {
    secs = (get_etime() - mystarttime)/(2.0*gip->iters);
    csecs = mycsecs / (2.0 * gip->iters);
  }

#if (defined(LOCK) || defined(REDUCE))
  spark_barrier();
#endif

  return NULL;
}

/*
 * assemble_matrix - assemble the local sparse matrix
 */
void assemble_matrix(double (*K)[DOF][DOF], struct gi *gip) {
  int i, j, k, l, m;
  int temp1, elem;

  for (i = 0; i < gip->matrixlen; i++) {
    for (j = 0; j < DOF; j++) {
      for (k = 0; k < DOF; k++) {
	K[i][j][k] = 0.0;
      }
    }
  }

  /*
   * add the contribution of each element to K
   */
  for (elem = 0; elem < gip->elems; elem++) {
    for (j = 0; j < gip->corners; j++) {
      for (k = 0; k < gip->corners; k++) {
	if (gip->vertex[elem][j] < gip->vertex[elem][k]) {
	  temp1 = gip->matrixindex[gip->vertex[elem][j]];
	  while (gip->matrixcol[temp1] != gip->vertex[elem][k]) {
	    temp1++;
	    if (temp1 >= gip->matrixindex[gip->vertex[elem][k] + 1]) {
	      fprintf(stderr, "%s: K indexing error in assemble %d %d\n",
		     gip->progname, temp1, gip->vertex[elem][k]);
	      exit(0);
	    }
	  }
	  for (l = 0; l < 3; l++) {
	    for (m = 0; m < 3; m++) {
	      K[temp1][l][m]++;
	    }
	  }
	}
      }
    }

  }

#ifdef DEBUGASSEMBLE
  local_printmatrix3(K, gip);
#endif
}

/*
 * assemble_vector - assemble the local v vector
 */
void assemble_vector(double (*v)[DOF], struct gi *gip) {
  int i, j;

  for (i = 0; i < gip->nodes; i++) {
    for (j = 0; j < DOF; j++) {
      v[i][j] = 1.0;
    }
  }

#ifdef DEBUGASSEMBLE
  local_printvec3(v, gip->nodes, gip);
#endif
}

/*
 * zero_vector - clear a portion of a vector
 */
void zero_vector(double (*v)[DOF], int firstrow, int numrows) {
  int i;
  for (i=firstrow; i<(firstrow+numrows); i++) {
    v[i][0] = 0.0;
    v[i][1] = 0.0;
    v[i][2] = 0.0;
  }
}

/*
 * local_smvp - local sparse matrix vector product w = K*v
 *    v and w are vectors of elements of size n x DOF.
 *    smvp is computed starting at row firstrow (zero based)
 *    and for numrows rows.
 *
 *    K is a block symmetric sparse matrix which is stored
 *    in compressed sparse row format as three vectors:
 *      - A is a coefficient vector of elements of size DOFxDOF. Only
 *        the upper right triangle is stored.
 *      - Acol[i] is the column number of the coefficient in K[i].
 *      - Row i consists of elements A[Acol[i]]...A[Acol[i+1]-1].
 */
void local_smvp(int nodes, double (*A)[DOF][DOF], int *Acol,
		int *Aindex, double (*v)[DOF], double (*w)[DOF],
		int firstrow, int numrows, int id, struct gi *gip) {
  int i;
  int Anext, Alast, col;
  double sum0, sum1, sum2;


#if defined(SCHEDULE)
    for (i = 0; i < schedule_len[id]; i++)
    {

        double elem[3][3];
        //printf("id = %d, i = %d \n", id, i);
        
        int i1, i2;
        for (i1 = 0; i1 < 3; i1++)
        {
            for (i2 = 0; i2 < 3; i2++)
            {
                //printf("i1 = %d, i2 = %d \n", i1, i2);
                //printf("%p \n", schedule[id][i]);
                
                elem[i1][i2] = (A[schedule[id][i]])[i1][i2];
                //printf("i1 = %d, i2 = %d \n", i1, i2);

            }
        }
        //printf("Segfault before \n");
        int r = schedule_row[id][i];

        sum0 = elem[0][0]*v[r][0] + elem[0][1]*v[r][1] + elem[0][2]*v[r][2];
        sum1 = elem[1][0]*v[r][0] + elem[1][1]*v[r][1] + elem[1][2]*v[r][2];
        sum2 = elem[2][0]*v[r][0] + elem[2][1]*v[r][1] + elem[2][2]*v[r][2];
        
        w[r][0] += sum0;
        w[r][1] += sum1;
        w[r][2] += sum2;
    }
#else
#if defined(LOCK)
  int lockid;
#endif

  for (i = firstrow; i < (firstrow + numrows); i++) {
    Anext = Aindex[i];
    Alast = Aindex[i + 1];

    sum0 = A[Anext][0][0]*v[i][0] + A[Anext][0][1]*v[i][1] + A[Anext][0][2]*v[i][2];
    sum1 = A[Anext][1][0]*v[i][0] + A[Anext][1][1]*v[i][1] + A[Anext][1][2]*v[i][2];
    sum2 = A[Anext][2][0]*v[i][0] + A[Anext][2][1]*v[i][1] + A[Anext][2][2]*v[i][2];

    Anext++;
    while (Anext < Alast) {
      col = Acol[Anext];

      sum0 += A[Anext][0][0]*v[col][0] + A[Anext][0][1]*v[col][1] + A[Anext][0][2]*v[col][2];
      sum1 += A[Anext][1][0]*v[col][0] + A[Anext][1][1]*v[col][1] + A[Anext][1][2]*v[col][2];
      sum2 += A[Anext][2][0]*v[col][0] + A[Anext][2][1]*v[col][1] + A[Anext][2][2]*v[col][2];

#if defined(LOCK)
      lockid = col % gip->locks;
      spark_setlock(lockid);
#endif

      w[col][0] += A[Anext][0][0]*v[i][0] + A[Anext][1][0]*v[i][1] + A[Anext][2][0]*v[i][2];
      w[col][1] += A[Anext][0][1]*v[i][0] + A[Anext][1][1]*v[i][1] + A[Anext][2][1]*v[i][2];
      w[col][2] += A[Anext][0][2]*v[i][0] + A[Anext][1][2]*v[i][1] + A[Anext][2][2]*v[i][2];

#if defined(LOCK)
      spark_unsetlock(lockid);
#endif

      Anext++;
    }

#if defined(LOCK)
    lockid = i % gip->locks;
    spark_setlock(lockid);
#endif

    w[i][0] += sum0;
    w[i][1] += sum1;
    w[i][2] += sum2;

#if defined(LOCK)
    spark_unsetlock(lockid);
#endif
  }
#endif
}

/*
 * add_vectors - add the vectors produced by each process
 */
void add_vectors(double (**tmpw)[DOF], double (*w)[DOF],
		 int firstrow, int numrows, struct gi *gip) {
  int i, j;
  double sum0, sum1, sum2;

  for (i = firstrow; i < (firstrow+numrows); i++) {
    sum0 = sum1 = sum2 = 0.0;
    for (j=0; j<gip->threads; j++) {
      sum0 += tmpw[j][i][0];
      sum1 += tmpw[j][i][1];
      sum2 += tmpw[j][i][2];
    }
    w[i][0] = sum0;
    w[i][1] = sum1;
    w[i][2] = sum2;
  }
}

/*
 * parsecommandline - read and interpret command line arguments
 */
void parsecommandline(int argc, char **argv, struct gi *gip) {
  int i, j;

  /* must have a file name */
  if (argc < 2) {
    usage(gip);
    exit(0);
  }

  /* first set up the defaults */
  gip->quiet = 0;
  gip->iters = DEFAULT_ITERS;
  gip->output = 0;
  gip->threads = 1;
  gip->locks = 1;

  /* now see if the user wants to change any of these */
  for (i=1; i<argc; i++) {
    if (argv[i][0] == '-') {
      if (argv[i][1] == 'i') {
	gip->iters = atoi(&argv[i][2]);
	if (gip->iters <= 0) {
	  fprintf(stderr, "error: iterations must be greater than zero.\n");
          fprintf(stderr, "no spaces allowed after the -i (e.g. -i100).\n");
	  exit(0);
	}
      }
#if (defined(LOCK) || defined(REDUCE))
      else if (argv[i][1] == 't') {
	gip->threads = atoi(&argv[i][2]);
	if (gip->threads <= 0) {
	  fprintf(stderr, "error: must have at least 1 thread.\n");
          fprintf(stderr, "no spaces allowed after the -t (e.g. -t8).\n");
	  exit(0);
	}
      }
#endif
#if defined(LOCK)
      else if (argv[i][1] == 'l') {
	gip->locks = atoi(&argv[i][2]);
	if (gip->locks < 0) {
	  fprintf(stderr, "error: must have at least 1 lock.\n");
          fprintf(stderr, "no spaces allowed after the -l (e.g. -l8).\n");
	  exit(0);
	}
      }
#endif

      else {
	for (j = 1; argv[i][j] != '\0'; j++) {
	  if (argv[i][j] == 'Q') {
	    gip->quiet = 1;
	  }
	  else if ((argv[i][j] == 'h' ||argv[i][j] == 'H')) {
	    info(gip);
	    exit(0);
	  }
	  else if (argv[i][j] == 'O') {
	    gip->output = 1;
	  }
	  else {
	    usage(gip);
	    exit(0);
	  }

	}
      }
    }
    else {
      strcpy(gip->packfilename, &argv[i][0]);
    }
  }
}

/*
 * info and usage - explain the command line arguments
 */
void info(struct gi *gip) {
  printf("\n");
  printf("You are running the %s kernel from the Spark98 Kernels.\n", gip->progname);
  printf("Copyright (C) 1998, David O'Hallaron, Carnegie Mellon University.\n");
  printf("You are free to use this software without restriction. If you find that\n");
  printf("the suite is helpful to you, it would be very helpful if you sent me a\n");
  printf("note at droh@cs.cmu.edu letting me know how you are using it.\n");
  printf("\n");
#if defined(LOCK)
  printf("%s is a parallel shared memory program based on locks. The threads update\n",
	 gip->progname);
  printf("a single output vector, using one or more lock to synchronize the updates.\n");
#elif defined(REDUCE)
  printf("%s is a parallel shared memory program based on reductions.\n",
	 gip->progname);
  printf("Each thread updates its own privatized output vector, which are\n");
  printf("later merged into a single output vector. There are no locks,\n");
  printf("and barriers are the only form of synchronization between threads.\n");
  printf("The style is similar to the Fx DO&MERGE model.\n");
#else
  printf("%s is the baseline sequential kernel.\n", gip->progname);
#endif
  printf("\n");

#if defined(LOCK)
  printf("%s [-hOQ] [-i<n>] [-l<n>] [-t<n>] packfilename\n\n", gip->progname);
#elif defined(REDUCE)
  printf("%s [-hOQ] [-i<n>] [-t<n>] packfilename\n\n", gip->progname);
#else
  printf("%s [-hOQ] [-i<n>] packfilename\n\n", gip->progname);
#endif
  printf("Command line options:\n\n");
  printf("    -h    Print this message and exit.\n");
  printf("    -i<n> Do n iterations of the SMVP pairs (default %d).\n",
	 DEFAULT_ITERS);
#if defined(LOCK)
  printf("    -l<n> Use n locks to sync the updates (default 1).\n");
#endif
  printf("    -O    Print the output vector to stdout.\n");
  printf("    -Q    Quietly suppress all explanations.\n");
#if (defined(LOCK) || defined(REDUCE))
  printf("    -t<n> Use n threads (default 1).\n");
#endif
  printf("\n");
  printf("Input packfiles are produced using the Archimedes tool chain. Packfiles\n");
  printf("for this program must consist of exactly 1 subdomain.\n");
  printf("\n");
#if defined(LOCK)
  printf("Example: %s -O -i10 -l256 -t8 sf5.1.pack\n", gip->progname);
#elif defined(REDUCE)
  printf("Example: %s -O -i10 -t8 sf5.1.pack\n", gip->progname);
#else
  printf("Example: %s -O -i10 sf5.1.pack\n", gip->progname);
#endif
}

void usage(struct gi *gip) {
  fprintf(stderr, "\n");
#if defined(LOCK)
  fprintf(stderr, "usage: %s [-hOQ] [-i<n>] [-l<n>] [-t<n>] packfilename\n\n", gip->progname);
#elif defined(REDUCE)
  fprintf(stderr, "usage: %s [-hOQ] [-i<n>] [-t<n>] packfilename\n\n", gip->progname);
#else
  fprintf(stderr, "usage: %s [-hOQ] [-i<n>] packfilename\n\n", gip->progname);
#endif
}


/*
 * readpackfile - reads and parses Archimedes packfile from disk
 */
void readpackfile(struct gi *gip) {
  int oldrow, newrow;
  int i, j, loop1;
  int temp1, temp2;
  int ival;
  FILE *packfile;
  char packfilename[STRLEN];

  if (!(packfile = fopen(gip->packfilename, "r"))) {
    fprintf(stderr, "%s: Can't open %s\n", gip->progname, gip->packfilename);
    exit(0);
  }

  /* read the basic size constants */
  fscanf(packfile, "%d", &gip->globalnodes);
  fscanf(packfile, "%d", &gip->mesh_dim);
  fscanf(packfile, "%d", &gip->globalelems);
  fscanf(packfile, "%d", &gip->corners);
  fscanf(packfile, "%d", &gip->subdomains);
  fscanf(packfile, "%d", &gip->processors);

  /* check for a valid packfile */
  if ((gip->subdomains < 1) || (gip->subdomains > 1024) ||
      ((gip->mesh_dim != 2) && (gip->mesh_dim != 3)) ||
      ((gip->corners != 3) && (gip->corners != 4)) ||
      gip->processors != gip->subdomains) {
    fprintf(stderr, "%s: the input file doesn't appear to be a packfile\n",
	    gip->progname);
    exit(0);
  }
  if (gip->subdomains != 1) {
    fprintf(stderr, "%s: You are using a packfile with %d subdomains. It should have exactly 1.\n",
	    gip->progname, gip->subdomains);
    exit(0);
  }

  /* read nodes */
  if (!gip->quiet) {
    fprintf(stderr, "%s: Reading nodes.", gip->progname);
  }

  fscanf(packfile, "%d %d %d", &gip->nodes, &gip->mine, &gip->priv);

  if (!(gip->coord = (void *)
	malloc(gip->nodes * gip->mesh_dim * sizeof(double)))) {
    fprintf(stderr, "%s: couldn't allocate coord(%d)\n",
	    gip->progname, gip->nodes);
    exit(0);
  }

  if (!(gip->globalnode = (int *) malloc(gip->nodes * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate globalnode(%d)\n",
	    gip->progname, gip->nodes);
    exit(0);
  }

  for (i=0; i<gip->nodes; i++) {
    fscanf(packfile, "%d", &gip->globalnode[i]);
    if (gip->globalnode[i] > gip->globalnodes || gip->globalnode[i] < 1) {
      fprintf(stderr,
	      "%s: bad node number(%d). Sure this is a packfile?\n",
	      gip->progname, gip->globalnode[i]);
	exit(0);
    }
    for (j=0; j<gip->mesh_dim; j++) {
      fscanf(packfile, "%lf", &gip->coord[i][j]);
    }
  }

  if (!gip->quiet) {
    fprintf(stderr, " Done.\n");
    fflush(stderr);
  }

  /* read elements */
  if (!gip->quiet)
    fprintf(stderr, "%s: Reading elements.", gip->progname);

  fscanf(packfile, "%d", &gip->elems);

  if (!(gip->vertex = (void *)
	malloc(gip->elems * gip->corners * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate vertex(%d)\n",
	    gip->progname, gip->elems);
    exit(0);
  }

  for (i=0; i<gip->elems; i++) {
    fscanf(packfile, "%d", &ival);
    for (j=0; j<gip->corners; j++) {
      fscanf(packfile, "%d", &gip->vertex[i][j]);
    }
  }

  if (!gip->quiet) {
    fprintf(stderr, " Done.\n");
    fflush(stderr);
  }

  /* read sparse matrix structure and convert from tuples to CSR */
  if (!gip->quiet)
    fprintf(stderr, "%s: Reading sparse matrix structure.", gip->progname);
  fscanf(packfile, "%d %d", &gip->matrixlen, &gip->choleskylen);

  /* allocate sparse matrix structures */
  if (!(gip->matrixindex = (int *) malloc((gip->nodes+1) * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate matrixindex(%d)\n",
	    gip->progname, gip->nodes+1);
    exit(0);
  }
  if (!(gip->matrixcol = (int *) malloc((gip->matrixlen+1) * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate matrixcol(%d)\n",
	    gip->progname, gip->matrixlen + 1);
    exit(0);
  }
  oldrow = -1;
  for (loop1 = 0; loop1 < gip->matrixlen; loop1++) {
    fscanf(packfile, "%d", &newrow);
    fscanf(packfile, "%d", &gip->matrixcol[loop1]);
    while (oldrow < newrow) {
      if (oldrow+1 >= gip->globalnodes+1) {
	printf("%s: error: (1)idx buffer too small (%d >= %d)\n",
	       gip->progname, oldrow+1, gip->globalnodes+1);
	exit(0);
      }
      gip->matrixindex[++oldrow] = loop1;
    }
  }
  while (oldrow < gip->globalnodes) {
    gip->matrixindex[++oldrow] = gip->matrixlen;
  }

  if (!gip->quiet) {
    fprintf(stderr, " Done.\n");
    fflush(stderr);
  }

  /* read null communication info (not relevant here) */
  fscanf(packfile, "%d %d", &temp1, &temp2);
}

/*
 * utility routines
 */
void printnodevector(double (*v)[3], int n) {
  int i;
  for (i=0; i<n; i++)
    printf("%d %.0f\n", i+1, v[i][0]);
  fflush(stdout);
}


/*
 * System-dependent timer routines
 */
#define MAX_ETIME 86400   /* 24 hour timer period */
struct itimerval first_u; /* user time */

/* init the timer */
void init_etime(void) {
  first_u.it_interval.tv_sec = 0;
  first_u.it_interval.tv_usec = 0;
  first_u.it_value.tv_sec = MAX_ETIME;
  first_u.it_value.tv_usec = 0;
  setitimer(ITIMER_VIRTUAL, &first_u, NULL);
}

/* return elapsed user seconds since call to init_etime */
double get_etime(void) {
  struct itimerval curr;

  getitimer(ITIMER_VIRTUAL, &curr);
  return (double) ((first_u.it_value.tv_sec - curr.it_value.tv_sec) +
		   (first_u.it_value.tv_usec - curr.it_value.tv_usec)*1e-6);
}

/*
 * System dependent thread routines
 */

/*
 * Posix thread (pthread) interface
 */
#if defined(PTHREAD)

void spark_init_threads() {
  int i, status;

  status = pthread_mutex_init(&barriermutexhandle, NULL);
  if (status != 0) {
    fprintf(stderr, "%s: couldn't initialize barrier mutex\n", gip->progname);
    exit(0);
  }

  status = pthread_cond_init(&barriercondhandle, NULL);
  if (status != 0) {
    fprintf(stderr, "%s: couldn't initialize barrier cond\n", gip->progname);
    exit(0);
  }

  if (!(vectormutexhandle = (pthread_mutex_t *)
	malloc(gip->locks * sizeof(pthread_mutex_t)))) {
    fprintf(stderr, "%s: couldn't allocate vectormutexhandle\n", gip->progname);
    exit(0);
  }
  for (i=0; i<gip->locks; i++) {
    status = pthread_mutex_init(&vectormutexhandle[i], NULL);
    if (status != 0) {
      fprintf(stderr, "%s: couldn't initialize vector mutex %d\n", gip->progname, i);
      exit(0);
    }
  }
}

void spark_start_threads(int n) {
  int i, status;
  pthread_t threadhandle;

  for (i=1; i<=n; i++) {
    ids[i] = i;
    status = pthread_create(&threadhandle, NULL, smvpthread, (void *)&ids[i]);
    if (status != 0) {
      fprintf(stderr, "%s: Could not create thread %d\n", gip->progname, i);
      exit(0);
    }
  }
}

void spark_barrier(void) {
  pthread_mutex_lock(&barriermutexhandle);
  barriercnt++;
  if (barriercnt == gip->threads) {
    barriercnt = 0;
    pthread_cond_broadcast(&barriercondhandle);
  }
  else {
    pthread_cond_wait(&barriercondhandle, &barriermutexhandle);
  }
  pthread_mutex_unlock(&barriermutexhandle);
}

void spark_setlock(int lockid) {
  pthread_mutex_lock(&vectormutexhandle[lockid]);
}

void spark_unsetlock(int lockid) {
  pthread_mutex_unlock(&vectormutexhandle[lockid]);
}
#endif

/*
 * SGI Threads Routines
 */
#if defined(SGI)
void spark_init_threads() {
  int i;

  arenahandle = usinit(arenaname);
  if (arenahandle == NULL) {
    fprintf(stderr, "arena init failed\n");
    exit(0);
  }

  barrierhandle = new_barrier(arenahandle);
  if (barrierhandle == NULL) {
    fprintf(stderr, "barrier init failed\n");
    exit(0);
  }

  if (!(lockhandle = (ulock_t *) malloc(gip->locks * sizeof(ulock_t)))) {
    fprintf(stderr, "%s: couldn't allocate lockhandle\n", gip->progname);
    exit(0);
  }
  for (i=0; i<gip->locks; i++) {
    lockhandle[i] = usnewlock(arenahandle);
    if (lockhandle[i] == NULL) {
      fprintf(stderr, "Couldn't create lock %d\n", i);
      exit(0);
    }
    usinitlock(lockhandle[i]);
  }
}

void spark_start_threads(int n) {
  int i;

  for (i=1; i<=n; i++) {
    ids[i] = i;
    (void)sproc((void (*)(void*)) smvpthread, PR_SADDR, (void *)&ids[i]);
  }
}

void spark_barrier(void) {
  barrier(barrierhandle, gip->threads);
}

void spark_setlock(int lockid) {
  ussetlock(lockhandle[lockid]);

}

void spark_unsetlock(int lockid) {
  usunsetlock(lockhandle[lockid]);
}
#endif

