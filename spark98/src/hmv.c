/***************************************************************************/
/* hmv.c -  hybrid SMVP kernel: a shared memory program in a time- and     */
/*             space-efficient message passing style                       */
/*                                                                         */
/* Spark98 Kernels                                                         */
/* Copyright (c) David O'Hallaron 1998                                     */
/*                                                                         */
/* Compilation options (you must define one of these two variables:        */
/*                                                                         */
/* Variable     Binary   Description                                       */
/* SGI          hmv      uses SGI thread primitives                        */
/* PTHREAD      hmv      uses Pthread thread primitives                    */
/*                                                                         */
/* You are free to use this software without restriction. If you find that */
/* the suite is helpful to you, it would be very helpful if you sent me a  */
/* note at droh@cs.cmu.edu letting me know how you are using it.           */
/***************************************************************************/ 

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#if defined(PTHREAD)
#include <pthread.h>
#elif defined(SGI)
#include <sys/types.h>
#include <sys/prctl.h>
#include <ulocks.h>
#endif

/* 
 * program wide constants 
 */
#define DEFAULT_ITERS 20  /* default number of SMVP operations */
#define STRLEN 128        /* default string length */
#define DOF 3             /* degrees of freedom in underlying simulation */
#define SEND_CMD 10       /* used by i/o master to ask for data from slaves */
#define INVALID -9999     /* marks uninitialized data */

/*
 * global variables
 */
struct gi { 
  /* misc */
  char progname[STRLEN];     /* program name */
  FILE *packfile;            /* file descriptor for pack file */

  /* command line options */
  int quiet;                 /* run quietly unless there are errors (-Q) */
  int iters;                 /* number of times to perform SMVP (-i<d>) */
  int output;                /* print the result vector? (-o) */
  char packfilename[STRLEN]; /* input packfile name */

  /* problem parameters */
  int globalnodes;           /* number of global nodes */
  int globalelems;           /* number of global elements */
  int mesh_dim;              /* mesh dimension (3) */
  int corners;               /* nodes per element (4) */
  int subdomains;            /* number of partition sets (and threads)*/ 
  int processors;            /* not used */
  int *nodes;                /* number of local nodes on each PE */
  int *mine;                 /* number of nodes that each PE owns */
  int *priv;                 /* number of unshared nodes per PE priv <= mine <= nodes */  
  int *elems;                /* number of elements on each PE */
  int *matrixlen;            /* number of nonzeros on each PE */
  int *friends;              /* number of PEs each PE communicates with */
  int *commlen;              /* number of communicated nodes for each PE */
  int maxcommlen;            /* max number of nodes shared by an PE */
  /* 
   * global data structures 
   */
  double (**coord)[DOF];     /* geometric node coordinates */
  int (**vertex)[4];         /* nodes associated with each element */
  int **globalnode;          /* local-to-global node mapping function */
  int **globalelem;          /* local-to-global element mapping function */

  /* sparse matrix in compressed sparse row format */
  int **matrixcol;           /* K[j] on PE i is in column matrixcol[i][j] */
  int **matrixindex;         /* row j on PE i starts at K[matrixindex[i][j]] */

  /* communication schedule */
  int **comm;                /* jth local node to send from PE i is comm[i][j] */
  int **commindex;           /* subdomain j on PE i starts at comm[i][commindex[j]] */

  /* tmp buffer for communication phase */
  double (**recvbuf)[DOF];

} gi, *gip=&gi;

/* timing variables */
double secs, csecs;

/* thread ids */
int *ids;

#if defined(PTHREAD)
/*
 * Pthread-specific thread variables
 */
pthread_mutex_t barriermutexhandle;
pthread_mutex_t vectormutexhandle;
pthread_cond_t barriercondhandle;
int barriercnt = 0;

#elif defined(SGI)
/* 
 * SGI-specific thread variables
 */
char arenaname[STRLEN]= "/dev/zero";
usptr_t *arenahandle;
barrier_t *barrierhandle;
ulock_t *lockhandle;
#endif

/*
 * SMVP data structures
 */

/* w1 = K1 * v1 */
double (**K1)[DOF][DOF];     /* sparse matrix */
double (**v1)[DOF];          /* dense vector */
double (**w1)[DOF];          /* dense vector */

/* w2 = K2 * v2 */
double (**K2)[DOF][DOF];     /* sparse matrix */
double (**v2)[DOF];          /* dense vector */
double (**w2)[DOF];          /* dense vector */

/* end globals */

/* 
 * function prototypes 
 */
/* initialization and exit routines */
void init(int argc, char **argv, struct gi *gip);
void parsecommandline(int argc, char **argv, struct gi *gip);
void readpackfile(struct gi *gip); 
void bail(struct gi *gip);
void finalize(struct gi *gip);

/* routines that read and parse the packfile */
void load_pack(struct gi *gip);
void load_global(struct gi *gip);
void load_nodes(struct gi *gip);
void load_elems(struct gi *gip);
void load_matrix(struct gi *gip);
void load_comm(struct gi *gip);

/* routine that assembles the sparse matrix and initial vector */
void assemble_matrix(double (**K)[DOF][DOF], struct gi *gip);
void assemble_vector(double (**v)[DOF], struct gi *gip);

/* sparse matrix vector multiply routines */
void *smvpthread(void *);
void zero_vector(double (*v)[DOF], int firstrow, int numrows);
void local_smvp(int nodes, double (*A)[DOF][DOF], int *Acol, int *Aindex, 
		double (*v)[DOF], double (*w)[DOF], int firstrow, int numrows);
void full_assemble(double (*v)[DOF], int id, struct gi *gip);

/* system dependent timer routines */
void init_etime(void);
double get_etime(void);

/* system dependent thread routines */
void spark_init_threads(void);
void spark_start_threads(int n);
void spark_barrier(void);
void spark_setlock(void);
void spark_unsetlock(void);

/* output routines */
void printnodevector(double (**v)[DOF], int n, struct gi *gip);

void par_printmatrix3(double (*A)[DOF][DOF], int id,struct gi *gip);
void par_printvec3(double (**v)[DOF], int n, int id, struct gi *gip);
void seq_printvec3(double (**v)[DOF], int id, struct gi *gip);

void seq_prglobal(struct gi *gip);
void seq_prnodes(struct gi *gip);
void seq_prelems(struct gi *gip);
void seq_prmatrix(struct gi *gip);
void seq_prcomm(struct gi *gip);
void seq_prcommvals(struct gi *gip);

/* misc routines */
void info(struct gi *gip);
void usage(struct gi *gip);


/*
 * main program 
 */
void main(int argc, char **argv) {
  int i, j;
  double mflops, global_mflops;
  int minnonzeros, maxnonzeros;

  init(argc, argv, gip);

  /* 
   * allocate storage for the local matrices and vectors
   */
  if (!(K1 = (void *)malloc((gip->subdomains*sizeof(double *))))) {
    fprintf(stderr, "%s: couldn't malloc K1\n", gip->progname);
    bail(gip);
  }
  if (!(K2 = (void *)malloc((gip->subdomains*sizeof(double *))))) {
    fprintf(stderr, "%s: couldn't malloc K2\n", gip->progname);
    bail(gip);
  }
  if (!(v1 = (void *)malloc((gip->subdomains*sizeof(double *))))) {
    fprintf(stderr, "%s: couldn't malloc v1\n", gip->progname);
    bail(gip);
  }
  if (!(v2 = (void *)malloc((gip->subdomains*sizeof(double *))))) {
    fprintf(stderr, "%s: couldn't malloc v2\n", gip->progname);
    bail(gip);
  }
  if (!(w1 = (void *)malloc((gip->subdomains*sizeof(double *))))) {
    fprintf(stderr, "%s: couldn't malloc w1\n", gip->progname);
    bail(gip);
  }
  if (!(w2 = (void *)malloc((gip->subdomains*sizeof(double *))))) {
    fprintf(stderr, "%s: couldn't malloc w2\n", gip->progname);
    bail(gip);
  }

  for (i=0; i<gip->subdomains; i++) {
    if (!(K1[i] = (void *)malloc((gip->matrixlen[i]+1) * DOF * DOF * sizeof(double)))) {
      fprintf(stderr, "%s: couldn't malloc K1[i](%d)\n", gip->progname);
      bail(gip);
    }
    if (!(K2[i] = (void *)malloc((gip->matrixlen[i]+1) * DOF * DOF * sizeof(double)))) {
      fprintf(stderr, "%s: couldn't malloc K2[i]\n", gip->progname);
      bail(gip);
    }
    if (!(v1[i] = (void *)malloc(gip->nodes[i] * DOF * sizeof(double)))) {
      fprintf(stderr, "%s: couldn't malloc v1\n", gip->progname);
      bail(gip);
    }
    if (!(v2[i] = (void *)malloc(gip->nodes[i] * DOF * sizeof(double)))) {
      fprintf(stderr, "%s: couldn't malloc v2\n", gip->progname);
      bail(gip);
    }
    if (!(w1[i] = (void *)malloc(gip->nodes[i] * DOF * sizeof(double)))) {
      fprintf(stderr, "%s: couldn't malloc w1\n", gip->progname);
      bail(gip);
    }
    if (!(w2[i] = (void *)malloc(gip->nodes[i] * DOF * sizeof(double)))) {
      fprintf(stderr, "%s: couldn't malloc w2\n", gip->progname);
      bail(gip);
    }
  }

  /* 
   * generate the sparse matrix coefficients 
   */
  if (!gip->quiet) 
    fprintf(stderr, "%s: Computing sparse matrix coefficients.", 
	    gip->progname);
  assemble_matrix(K1, gip);
  assemble_matrix(K2, gip);
  assemble_vector(v1, gip);
  assemble_vector(v2, gip);

  if (!gip->quiet) {
    fprintf(stderr, " Done.\n");
    fflush(stderr);
  }

  /* 
   * start the workers
   */
  spark_start_threads(gip->subdomains-1);

  /*
   * start the master 
   */
  if (!gip->quiet) {
    fprintf(stderr, "%s: Performing %d SMVP pairs with %d threads.", 
	    gip->progname, gip->iters, gip->subdomains);
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
    bail(gip);
  }

  /* 
   * Summarize performance 
   */
  mflops = 0.0;
  for (i=0; i<gip->subdomains; i++) {
    mflops += 
      (double)((2.0*gip->matrixlen[i] - gip->nodes[i])  /* nonzero blocks */
	       *DOF*DOF                                 /* DOF^2 numbers/block */
	       *2.0)                                    /* 2 flops/block */
      / 1000000.0;
  }    

  /* find the minimum and maximum load on each subdomain */ 
  minnonzeros = 1<<20;
  maxnonzeros = -1;
  for (i=0; i<gip->subdomains; i++) {
    if (gip->matrixlen[i] < minnonzeros) 
      minnonzeros = gip->matrixlen[i];
    if (gip->matrixlen[i] > maxnonzeros) 
      maxnonzeros = gip->matrixlen[i];
  }

    fprintf(stderr, 
	    "%s: %s %.6f Mf %.6f s [%.6f/%.6f/%.0f%%] %.1f Mf/s [%d/%d/%.0f%%]\n",
	    gip->progname, gip->packfilename, mflops, secs, 
	    secs - csecs, csecs, 
	    ((secs-csecs)/secs)*100.0, 
	    mflops/secs,
	    minnonzeros, maxnonzeros, 
	    (double)((double)minnonzeros/(double)maxnonzeros)*100.0);
  

  /* 
   * print results if asked for
   */
  if (gip->output) {
    printnodevector(w1, gip->globalnodes, gip);
  }

  /*
   * Clean up
   */
  if (!gip->quiet) {
    fprintf(stderr, "%s: Done.\n", gip->progname);
  }
  finalize(gip);
}

/*
 * smvpthread - thread that performs a sequence of SMVP pairs 
 */
void *smvpthread(void *a) {
  int i, j;
  double mycsecs, mystarttime, mystartctime;
  int *argp = (int *)a;
  int id = *argp;

  spark_barrier();

  mycsecs = 0.0;
  mystarttime = get_etime();

  for (i=0; i<gip->iters; i++) {

    /* w1 = K1 * v1 */
    zero_vector(w1[id], 0, gip->nodes[id]);
    local_smvp(gip->nodes[id], K1[id], gip->matrixcol[id], 
	       gip->matrixindex[id], v1[id], w1[id], 0, gip->nodes[id]);
    mystartctime = get_etime();
    full_assemble(w1[id], id, gip);
    mycsecs += (get_etime() - mystartctime);
    
    /* w2 = K2 * v2 */
    zero_vector(w2[id], 0, gip->nodes[id]);
    local_smvp(gip->nodes[id], K2[id], gip->matrixcol[id], 
	       gip->matrixindex[id], v2[id], w2[id], 0, gip->nodes[id]);
    mystartctime = get_etime();
    full_assemble(w2[id], id, gip);
    mycsecs += (get_etime() - mystartctime);
  }

  if (id == 0) {
    secs = (get_etime() - mystarttime)/(gip->iters*2.0);
    csecs = mycsecs / (gip->iters*2.0);
  }
  return NULL;
}

/*
 * assemble_matrix - assemble the local sparse matrix
 */
void assemble_matrix(double (**K)[DOF][DOF], struct gi *gip) {
  int i, j, k, l, m, s;
  int temp1, elem;

  for (s=0; s<gip->subdomains; s++) {
    for (i = 0; i < gip->matrixlen[s]; i++) {
      for (j = 0; j < DOF; j++) {
	for (k = 0; k < DOF; k++) {
	  K[s][i][j][k] = 0.0;
	}
      }
    }
  }

  /* 
   * add the contribution of each element to K
   */
  for (s=0; s<gip->subdomains; s++) {
    for (elem = 0; elem < gip->elems[s]; elem++) {
      for (j = 0; j < gip->corners; j++) {
	for (k = 0; k < gip->corners; k++) {
	  if (gip->vertex[s][elem][j] < gip->vertex[s][elem][k]) {
	    temp1 = gip->matrixindex[s][gip->vertex[s][elem][j]];
	    while (gip->matrixcol[s][temp1] != gip->vertex[s][elem][k]) {
	      temp1++;
	      if (temp1 >= gip->matrixindex[s][gip->vertex[s][elem][k] + 1]) {
		fprintf(stderr, "%s: K indexing error in assemble %d %d\n", 
			gip->progname, temp1, gip->vertex[s][elem][k]);
		bail(gip);
	      }
	    }
	    for (l = 0; l < 3; l++) {
	      for (m = 0; m < 3; m++) {
		K[s][temp1][l][m] += 1.0;
	      }
	    }
	  }
	}
      }
    }
  } 

#ifdef DEBUGASSEMBLE
  seq_printmatrix3(K[i], gip);
#endif
}

/*
 * assemble_vector - assemble the local v vector
 */
void assemble_vector(double (**v)[DOF], struct gi *gip) {
  int i, j, s;

  for (s=0; s<gip->subdomains; s++) {
    for (i = 0; i < gip->nodes[s]; i++) {
      for (j = 0; j < DOF; j++) {
	v[s][i][j] = 1.0;
      }
    } 
  }

#ifdef DEBUGASSEMBLE
  seq_printvec3(v, gip);
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
		int firstrow, int numrows) {
  int i;
  int Anext, Alast, col;
  double sum0, sum1, sum2;
  
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
      
      w[col][0] += A[Anext][0][0]*v[i][0] + A[Anext][1][0]*v[i][1] + A[Anext][2][0]*v[i][2];
      w[col][1] += A[Anext][0][1]*v[i][0] + A[Anext][1][1]*v[i][1] + A[Anext][2][1]*v[i][2];
      w[col][2] += A[Anext][0][2]*v[i][0] + A[Anext][1][2]*v[i][1] + A[Anext][2][2]*v[i][2];

      Anext++;
    }

    w[i][0] += sum0;
    w[i][1] += sum1;
    w[i][2] += sum2;
  }
}

/*
 * full_assemble - assemble a distributed vector
 */
void full_assemble(double (*v)[DOF], int id, struct gi *gip) {
  int i, j, s, pos;
  int overwrite;

  spark_barrier();
  
  /* copy partial results to our neighbors' receive buffers */
  for (s = 0; s < gip->subdomains; s++) {
    if (gip->commindex[id][s] < gip->commindex[id][s+1]) {
      pos = gip->commindex[s][id];
      for (j = gip->commindex[id][s]; j < gip->commindex[id][s+1]; j++) {
	gip->recvbuf[s][pos][0] = v[gip->comm[id][j]][0];
	gip->recvbuf[s][pos][1] = v[gip->comm[id][j]][1];
	gip->recvbuf[s][pos++][2] = v[gip->comm[id][j]][2];
      }
    }
  }

  spark_barrier();
  
  /* update the copies of my node variables */
  for (i = 0; i < gip->commlen[id]; i++) {
    v[gip->comm[id][i]][0] += gip->recvbuf[id][i][0];
    v[gip->comm[id][i]][1] += gip->recvbuf[id][i][1];
    v[gip->comm[id][i]][2] += gip->recvbuf[id][i][2];
  }

  spark_barrier();
}

void init(int argc, char **argv, struct gi *gip) {
  int i;
  char *sp;    

  init_etime();
  spark_init_threads();

  /* 
   * no need to see the entire path name, the base will do. 
   */ 
  for (sp = argv[0]+strlen(argv[0]); (sp != argv[0]) && *sp != '/'; sp--)
    ;
  if (*sp == '/')
    strcpy(gip->progname, sp+1);
  else    
    strcpy(gip->progname, sp);

  parsecommandline(argc, argv, gip);

  if (!(gip->packfile = fopen(gip->packfilename, "r"))) {
    fprintf(stderr, "%s: Can't open %s\n", gip->progname, gip->packfilename);
    bail(gip);
  }

  load_pack(gip);

  if (!(gip->recvbuf = (void *)malloc(gip->subdomains * sizeof(double *)))) {
    fprintf(stderr, "%s: couldn't allocate recvbuf\n", gip->progname);
    bail(gip);
  }
  for (i=0; i<gip->subdomains; i++) {
    if (!(gip->recvbuf[i] = (void *)malloc((gip->maxcommlen+1)*DOF*sizeof(double)))) {
      fprintf(stderr, "%s: couldn't allocate recvbuf[i]\n", gip->progname);
      bail(gip);
    }
  }

  if (!(ids = (int *)malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate ids\n", gip->progname);
    bail(gip);
  }
}

/*
 * load_pack - load the input packfile
 */
void load_pack(struct gi *gip) {
  if (!(gip->packfile = fopen(gip->packfilename, "r"))) {
    fprintf(stderr, "%s: Can't open %s\n", 
	    gip->progname, gip->packfilename);
    bail(gip);
  }
  
  load_global(gip);
  load_nodes(gip);
  load_elems(gip);
  load_matrix(gip);
  load_comm(gip);
}

void load_global(struct gi *gip) {
  int ibuf[6];
  
  fscanf(gip->packfile, "%d %d\n", &gip->globalnodes, &gip->mesh_dim);
  fscanf(gip->packfile, "%d %d\n", &gip->globalelems, &gip->corners);
  fscanf(gip->packfile, "%d\n", &gip->subdomains);
  fscanf(gip->packfile, "%d\n", &gip->processors);

  if ((gip->subdomains < 1) || (gip->subdomains > 1024) ||
      ((gip->mesh_dim != 2) && (gip->mesh_dim != 3)) ||
      ((gip->corners != 3) && (gip->corners != 4)) ||
      gip->processors != gip->subdomains) {
    fprintf(stderr, "%s: the input file doesn't appear to be a packfile\n",
	    gip->progname);
    bail(gip);
  }

#ifdef DEBUGGLOBAL    
  seq_prglobal(gip);
#endif
}

void load_nodes(struct gi *gip) {
  int i, j, k;

  if (!gip->quiet) {
    fprintf(stderr, "%s: Reading nodes", gip->progname);
    fflush(stderr);
  }
  if (!(gip->nodes = (int *) malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate nodes\n", 
	    gip->progname);
    bail(gip);
  }
  if (!(gip->priv = (int *) malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate nodepriv\n", gip->progname);
    bail(gip);
  }
  if (!(gip->mine = (int *) malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate mine\n", gip->progname);
    bail(gip);
  }

  for (i = 0; i < gip->subdomains; i++) {
    fscanf(gip->packfile, "%d %d %d\n", &gip->nodes[i], &gip->mine[i], &gip->priv[i]);
  }

  if (!(gip->coord = (void *) malloc(gip->subdomains*sizeof(double *)))) {
    fprintf(stderr, "%s: couldn't allocate coord\n", gip->progname);
    bail(gip);
  }
  for (i=0; i<gip->subdomains; i++) {
    if (!(gip->coord[i] = (void *) malloc(gip->nodes[i]*gip->mesh_dim*sizeof(double)))) {
      fprintf(stderr, "%s: couldn't allocate coord[i]\n", gip->progname);
      bail(gip);
    }
  }

  if (!(gip->globalnode = (void *) malloc(gip->subdomains * sizeof(int *)))) {
    fprintf(stderr, "%s: couldn't alloc globalnode\n", gip->progname);
    bail(gip);
  }
  for (i=0; i<gip->subdomains; i++) {
    if (!(gip->globalnode[i] = (int *) malloc(gip->nodes[i]*sizeof(int)))) {
      fprintf(stderr, "%s: couldn't alloc globalnode[i]\n", gip->progname);
      bail(gip);
    }
  }

  for (i = 0; i < gip->subdomains; i++) {
    if (!gip->quiet) {
      fprintf(stderr, ".");
      fflush(stderr);
    }

    /* node data */
    for (j = 0; j < gip->nodes[i]; j++) {
      fscanf(gip->packfile, "%d", &gip->globalnode[i][j]);
      for (k = 0; k < gip->mesh_dim; k++) {
	fscanf(gip->packfile, "%lf", &gip->coord[i][j][k]);
      }
    }
  }

  if (!gip->quiet) {
    fprintf(stderr, "done\n");
    fflush(stderr);
  }

#ifdef DEBUGNODES
  seq_prnodes(gip);
#endif
}

void load_elems(struct gi *gip) {
  int i, j, k;

  if (!gip->quiet) {
    fprintf(stderr, "%s: Reading elements", gip->progname);
    fflush(stderr);
  }
  if (!(gip->elems = (int *) malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate elems\n", gip->progname);
    bail(gip);
  }

  for (i = 0; i < gip->subdomains; i++) {
    fscanf(gip->packfile, "%d\n", &gip->elems[i]);
  }
    
  if (!(gip->vertex = (void *) malloc(gip->subdomains * sizeof(int *)))) {
    fprintf(stderr, "%s: couldn't allocate vertex\n", gip->progname);
    bail(gip);
  }
  for (i=0; i<gip->subdomains; i++) {
    if (!(gip->vertex[i] = (void *) malloc(gip->elems[i] * gip->corners * sizeof(int)))) {
      fprintf(stderr, "%s: couldn't allocate vertex[i]\n", gip->progname);
      bail(gip);
    }
  }

  if (!(gip->globalelem = (void *) malloc(gip->subdomains * sizeof(int *)))) {
    fprintf(stderr, "%s: couldn't alloc globalelem\n", gip->progname);
    bail(gip);
  }
  for (i=0; i<gip->subdomains; i++) {
    if (!(gip->globalelem[i] = (void *) malloc(gip->elems[i] * sizeof(int)))) {
      fprintf(stderr, "%s: couldn't alloc globalelem\n", gip->progname);
      bail(gip);
    }
  }

  for (i = 0; i < gip->subdomains; i++) {

    if (!gip->quiet) {
      fprintf(stderr, ".");
      fflush(stderr);
    }

    for (j = 0; j < gip->elems[i]; j++) {
      fscanf(gip->packfile, "%d", &gip->globalelem[i][j]);
      for (k = 0; k < gip->corners; k++) {
	fscanf(gip->packfile, "%d", &gip->vertex[i][j][k]);
      }
    }
  }

  if (!gip->quiet) {
    fprintf(stderr, "done\n");
    fflush(stderr);
  }

#ifdef DEBUGELEMS
  seq_prelems(gip);
#endif
}  

void load_matrix(struct gi *gip) {
  int i, k, loop1;
  int oldrow, newrow;

  if (!gip->quiet) {
    fprintf(stderr, "%s: Reading matrix", gip->progname);
    fflush(stderr);
  }

  if (!(gip->matrixlen = (void *) malloc(gip->subdomains * sizeof(int *)))) {
    fprintf(stderr, "%s: couldn't allocate matrixlen\n", gip->progname);
    bail(gip);
  }

  /* get the number of i,j pairs on each subdomain */
  for (i = 0; i < gip->subdomains; i++) { 
    fscanf(gip->packfile, "%d %*d\n", &gip->matrixlen[i]); 
  }

  if (!(gip->matrixcol = (void *) malloc(gip->subdomains * sizeof(int *)))) {
    fprintf(stderr, "%s: couldn't allocate matrixcol\n", gip->progname);
    bail(gip);
  }
  for (i=0; i<gip->subdomains; i++) {
    if (!(gip->matrixcol[i] = (void *) malloc((gip->matrixlen[i]+1) * sizeof(int)))) {
      fprintf(stderr, "%s: couldn't allocate matrixcol[i]\n", gip->progname);
      bail(gip);
    }
  }

  if (!(gip->matrixindex = (void *) malloc(gip->subdomains * sizeof(int *)))){
    fprintf(stderr, "%s: couldn't allocate matrixindex\n", gip->progname);
    bail(gip);
  }
  for (i=0; i<gip->subdomains; i++) {
    if (!(gip->matrixindex[i] = (int *) malloc((gip->nodes[i]+1) * sizeof(int)))){
      fprintf(stderr, "%s: couldn't allocate matrixindex\n", gip->progname);
      bail(gip);
    }
  }
    
  for (i = 0; i < gip->subdomains; i++) {

    if (!gip->quiet) {
      fprintf(stderr, ".");
      fflush(stderr);
    }

    oldrow = -1; 
    for (loop1 = 0; loop1 < gip->matrixlen[i]; loop1++) {

      fscanf(gip->packfile, "%d", &newrow);
      fscanf(gip->packfile, "%d", &gip->matrixcol[i][loop1]);
      while (oldrow < newrow) {
	if (oldrow+1 >= gip->matrixlen[i]) {
	  fprintf(stderr, "%s: index buffer(1) too small (%d >= %d)\n", 
		  gip->progname, oldrow+1, gip->matrixlen[i]);
	  bail(gip);
	}
	gip->matrixindex[i][++oldrow] = loop1;
      }
    }
    while (oldrow < gip->nodes[i]) {
      if (oldrow+1 >= gip->matrixlen[i]) {
	fprintf(stderr, "%s: index buffer(2) too small (%d >= %d)\n", 
		gip->progname, oldrow+1, gip->matrixlen[i]);
	bail(gip);
      }
      gip->matrixindex[i][++oldrow] = gip->matrixlen[i];
    }

  }

  if (!gip->quiet) {
    fprintf(stderr, "done\n");
    fflush(stderr);
  }

#ifdef DEBUGMATRIX
  seq_prmatrix(gip);
#endif
}

void load_comm(struct gi *gip) {
  int i, j, k;
  int count;
  int *commnodes; /*subdomain i shares a total of commnodes[i] nodes */
  int *buf;
  
  if (!gip->quiet) {
    fprintf(stderr, "%s: Reading communication info", gip->progname);
    fflush(stderr);
  }
  
  if (!(gip->friends = (int *) malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate friends\n", gip->progname);
    bail(gip);
  }

  if (!(gip->commlen = (int *) malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate commlen\n", gip->progname);
    bail(gip);
  }

  if (!(commnodes = (int *) malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate commnodes\n", gip->progname);
    bail(gip);
  }
  

  gip->maxcommlen = 0;
  for (i = 0; i < gip->subdomains; i++) {
    fscanf(gip->packfile, "%d\n", &commnodes[i]);
    if (commnodes[i] > gip->maxcommlen)
      gip->maxcommlen = commnodes[i];
  }
  
  if (!(gip->comm = (void *) malloc(gip->subdomains * sizeof(int *)))) {
    fprintf(stderr, "%s: couldn't allocate comm\n", gip->progname);
    bail(gip);
  }
  for (i=0; i<gip->subdomains; i++) {
    if (!(gip->comm[i] = (int *) malloc((commnodes[i]+1) * sizeof(int)))) {
      fprintf(stderr, "%s: couldn't allocate comm[i]\n", gip->progname);
      bail(gip);
    }
  }

  if (!(gip->commindex = (void *) malloc(gip->subdomains * sizeof(int *)))) {
    fprintf(stderr, "%s: couldn't allocate commindex\n", gip->progname);
    bail(gip);
  }

  for (i=0; i<gip->subdomains; i++) {
    if (!(gip->commindex[i] = (int *) malloc((gip->subdomains+1) * sizeof(int)))) {
      fprintf(stderr, "%s: couldn't allocate commindex\n", gip->progname);
      bail(gip);
    }
  }

  for (i = 0; i < gip->subdomains; i++) {
    
    if (!gip->quiet) {
      fprintf(stderr, ".");
      fflush(stderr);
    }
    
    gip->commlen[i] = 0;
    gip->friends[i] = 0;

    for (j = 0; j < gip->subdomains; j++) {
      
      gip->commindex[i][j] = gip->commlen[i];

      /* subdomain i shares count nodes with subdomain j */
      fscanf(gip->packfile, "%d", &count);
      if (count > commnodes[i]) {
	fprintf(stderr, "%s: count exceeds commnodes[i]\n", gip->progname);
	bail(gip);
      }
      
      if (count > 0)
	gip->friends[i]++;

      for (k = 0; k < count; k++) {
	fscanf(gip->packfile, "%d", &gip->comm[i][gip->commlen[i]++]);
      }
    }
    gip->commindex[i][gip->subdomains] = gip->commlen[i];

    if (gip->commlen[i] != commnodes[i]) {
      fprintf(stderr, "%s: inconsistent comm lengths\n", gip->progname);
      bail(gip);
    }
  }

  (void)free(commnodes);
  
  if (!gip->quiet) {
    fprintf(stderr, "done\n");
    fflush(stderr);
  }

#ifdef DEBUGCOMM
    seq_prcomm(gip);
#endif
}    

/*
 * parsecommandline - read and interpret command line arguments
 */
void parsecommandline(int argc, char **argv, struct gi *gip) {
  int i, j;
    
  /* must have a file name */
  if (argc < 2) {
    usage(gip);
  }

  /* first set up the defaults */
  gip->quiet = 0;
  gip->iters = DEFAULT_ITERS;
  gip->output = 0;

  /* now see if the user wants to change any of these */
  for (i=1; i<argc; i++) {
    if (argv[i][0] == '-') {
      if (argv[i][1] == 'i') {
	gip->iters = atoi(&argv[i][2]);
	if (gip->iters <= 0) {
	  fprintf(stderr, "error: iterations must be greater than zero.\n");
          fprintf(stderr, "no spaces allowed after the -i (e.g. -i100).\n");
	  bail(gip);
	}
      }
      else {
	for (j = 1; argv[i][j] != '\0'; j++) {
	  if (argv[i][j] == 'Q') {
	    gip->quiet = 1;
	  }
	  else if ((argv[i][j] == 'h' ||argv[i][j] == 'H')) {
	    info(gip);
	    finalize(gip);
	  }
	  else if (argv[i][j] == 'O') {
	    gip->output = 1;
	  }
	  else {
	    usage(gip);
	    bail(gip);
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
  printf("You are running the %s kernel from the Spark98 Benchmarks.\n", 
	 gip->progname);
  printf("Copyright (C) 1998, David O'Hallaron, Carnegie Mellon University.\n");
  printf("You are free to use this software without restriction. If you find that\n");
  printf("the suite is helpful to you, it would be very helpful if you sent me a\n");
  printf("note at droh@cs.cmu.edu letting me know how you are using it.\n"); 
  printf("\n");
  printf("%s is a hybrid program based on shared memory but written in a\n", gip->progname);
  printf("message passing style. Each thread performs its own local SMVP\n");
  printf("operation using partially assembled local matrices and vectors\n");
  printf("followed by an assembly phase that combines the partially assembled\n");
  printf("output vectors.\n");
  printf("\n");
  printf("%s [-hOQ] [-i<n>] packfilename\n\n", gip->progname);
  printf("Command line options:\n\n");
  printf("    -h    Print this message and exit.\n");
  printf("    -i<n> Do n iterations of the SMVP pairs (default %d).\n",
	 DEFAULT_ITERS);
  printf("    -O    Print the output vector to stdout.\n");
  printf("    -Q    Quietly suppress all explanations.\n");
  printf("\n");
  printf("Input packfiles are produced using the Archimedes tool chain.\n");
  printf("\n");
  printf("Example: %s -O -i10 sf5.8.pack\n", gip->progname);
}

void usage(struct gi *gip) {
  fprintf(stderr, "\n");
  fprintf(stderr, "usage: %s [-OQh] [-i<n>] packfilename\n\n", gip->progname);
  exit(0);
}

/* 
 * sequential output routines
 */

/* 
 * printnodevector - print one column of a global nodevector
 */
void printnodevector(double (**v)[DOF], int n, struct gi *gip) {
  int i, j, k, found;

  for (k=1; k<=n; k++) {
    found = 0;
    for (i=0; i<gip->subdomains && !found; i++) {
      for (j=0; j<gip->nodes[i]; j++) {
	if (k == gip->globalnode[i][j]) {
	  printf("%d %.0f\n", k, v[i][j][0]);
	  found = 1;
	  break;
	}
      }
    }
  }
}

/* 
 * printvec3 - print the local version of a vector on each subdomain
 */
void printvec3(double (**v)[DOF], int id, struct gi *gip) {
  int i, s;
  
  for (s=0; s<gip->subdomains; s++) {
    for (i = 0; i < gip->nodes[id]; i++) {
      printf("[%d]: %d (%d): %.0f\n", s, i, 
	     gip->globalnode[id][i], v[id][i][0]);
    }
    fflush(stdout);
  }
}


/*
 * parallel output routines
 */

/* emit local matrix in order on each subdomain */
void par_printmatrix3(double (*A)[DOF][DOF], int id, struct gi *gip) {
  int i, j, k, l, row, col, s;

  for (s=0; s<gip->subdomains; s++) {
    spark_barrier();
    if (id == s) {
      for (i = 0; i < gip->nodes[s]; i++) { 
	for (j = 0; j < 3; j++) {
	  for (k = gip->matrixindex[s][i]; k < gip->matrixindex[s][i+1]; k++) {
	    for (l = 0; l < 3; l++) {
	      row =  i*3 + j + 1; /*row*/
	      col =  gip->matrixcol[s][k]*3 + l + 1; /*col*/
	      printf("%d: %d %d %.0f\n", s, row, col, A[k][j][l]);
	      printf("%d: %d %d %.0f\n", s, col, row, A[k][j][l]);
	    }
	  }
	}
      }
    }
  }
}

/* emit local vector in order on each subdomain */
void par_printvec3(double (**v)[DOF], int n, int id, struct gi *gip) {
  int i, s;
  
  for (s=0; s<gip->subdomains; s++) {
    spark_barrier();
    if (id == s) {
      printf("%d: ready to print\n", id);
      for (i = 0; i < n; i++) {
	printf("[%d]: %d (%d): %.0f\n", id, i, 
	       gip->globalnode[s][i], v[s][i][0]);
      }
      fflush(stdout);
    }
  }
}

/*
 * sequential debug routines for the pack file input
 */

/* print global info */
void seq_prglobal(struct gi *gip) {
  int i;

  printf("gnodes=%d dim=%d gelem=%d corners=%d subdomains=%d procs=%d\n",
	 gip->globalnodes, gip->mesh_dim, 
	 gip->globalelems, gip->corners, gip->subdomains, 
	 gip->processors);
  fflush(stdout);
}

/* print finite element mesh nodes on each subdomain */
void seq_prnodes(struct gi *gip) {
  int i, j, k; 
  
  for (i = 0; i < gip->subdomains; i++) {
    printf("[%d]: gip->nodes=%d gip->priv=%d gip->mine=%d\n", 
	   i, gip->nodes[i], gip->priv[i], gip->mine[i]);
    for (j = 0; j < gip->nodes[i]; j++) {
      printf("[%d] %d (%d): ", i, j, gip->globalnode[i][j]);
      for (k = 0; k < gip->mesh_dim; k++) {
	printf("%f ", gip->coord[i][j][k]);
      }
      printf("\n");	
    }
    fflush(stdout);
  }
}

/* print finite element mesh elements on each subdomain */
void seq_prelems(struct gi *gip)
{
  int i, j, k;
    
  for (i = 0; i < gip->subdomains; i++) {
    printf("[%d]: gip->elems=%d\n", i, gip->elems[i]);  
    for (j = 0; j < gip->elems[i]; j++) {
      printf("[%d]: %d (%d): ", i, j, gip->globalelem[i][j]);
      for (k = 0; k < gip->corners; k++) {
	printf("%d ", gip->vertex[i][j][k]);
      }
      printf("\n");
      fflush(stdout);
    }
  }
}

/* print sparse matrix  structure on each subdomain */
void seq_prmatrix(struct gi *gip)
{
  int i, j;
    
  for (i = 0; i < gip->subdomains; i++) {
    fflush(stdout);
    printf("[%d]: gip->matrixcol:\n", i);	
    for (j = 0; j < gip->matrixlen[i]; j++)
      printf("[%d]: %d: %d\n", i, j, gip->matrixcol[i][j]); 
    printf("[%d]: gip->matrixindex:\n", i);	
    for (j = 0; j <= gip->nodes[i]; j++)
      printf("[%d]: %d: %d\n", i, j, gip->matrixindex[i][j]);
    fflush(stdout);
  }
}

/* print communication schedule on each subdomain */
void seq_prcomm(struct gi *gip) {
  int i, j, k;
  
  for (i = 0; i < gip->subdomains; i++) {
    printf("[%d]: comm info (%d shared nodes, %d friends)\n", 
	   i, gip->commlen[i], gip->friends[i]);
    for (j = 0; j < gip->subdomains; j++) {
      printf("[%d]: %d nodes shared with subdomain %d:\n", 
	     i, gip->commindex[i][j+1]-gip->commindex[i][j],j);
      fflush(stdout);
      for (k = gip->commindex[i][j]; k < gip->commindex[i][j+1]; k++) {
	printf("[%d]: %d %d (%d)\n", 
	       i, k, gip->comm[i][k], gip->globalnode[i][gip->comm[i][k]]);
	fflush(stdout);
      }
    }
    fflush(stdout);
  }
}

/* print communication schedule with values of the receive buffer */
void seq_prcommvals(struct gi *gip) {
  int i, j, k;
  
  for (i = 0; i < gip->subdomains; i++) {
    printf("[%d]: comm info (%d shared nodes, %d friends)\n", 
	   i, gip->commlen[i], gip->friends[i]);
    for (j = 0; j < gip->subdomains; j++) {
      printf("[%d]: %d nodes shared with subdomain %d:\n", 
	     i, gip->commindex[i][j+1]-gip->commindex[i][j],j);
      fflush(stdout);
      for (k = gip->commindex[i][j]; k < gip->commindex[i][j+1]; k++) {
	printf("[%d]: %d %d (%d) recvbuf[%d][%d]=%.0f\n", 
	       i, k, gip->comm[i][k], gip->globalnode[i][gip->comm[i][k]],
	       i, k, gip->recvbuf[i][k][0]);
	fflush(stdout);
      }
    }
    fflush(stdout);
  }
}

/*
 * termination routines 
 */

/* orderly exit */
void finalize(struct gi *gip) {
  if (!gip->quiet) {
    fprintf(stderr, "%s: Terminating normally.\n", gip->progname);
    fflush(stderr);
  }

  exit(0);
}

/* emergency exit */
void bail(struct gi *gip) {
  fprintf(stderr, "\n");
  fprintf(stderr, "%s: Something bad happened. Terminating abnormally.\n", 
	  gip->progname);
  fprintf(stderr, "\n");
  fflush(stderr);
  fflush(stdout);
  exit(0);
}


/*
 * system dependent timer routines
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
 * Posix Threads Routines (pthread)
 */
#if defined(PTHREAD)
void spark_init_threads() {
  int status; 

  status = pthread_mutex_init(&barriermutexhandle, NULL);
  if (status != 0) {
    fprintf(stderr, "%s: couldn't initialize barrier mutex\n", gip->progname);
    exit(0);
  }
  status = pthread_mutex_init(&vectormutexhandle, NULL);
  if (status != 0) {
    fprintf(stderr, "%s: couldn't initialize vector mutex\n", gip->progname);
    exit(0);
  }
  status = pthread_cond_init(&barriercondhandle, NULL);
  if (status != 0) {
    fprintf(stderr, "%s: couldn't initialize barrier cond\n", gip->progname);
    exit(0);
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
  if (barriercnt == gip->subdomains) {
    barriercnt = 0;
    pthread_cond_broadcast(&barriercondhandle);
  }
  else {
    pthread_cond_wait(&barriercondhandle, &barriermutexhandle);
  }
  pthread_mutex_unlock(&barriermutexhandle);
}

void spark_setlock() {
  pthread_mutex_lock(&vectormutexhandle);
}

void spark_unsetlock() {
  pthread_mutex_unlock(&vectormutexhandle);
}
#endif

/*
 * SGI Threads Routines 
 */
#if defined(SGI)
void spark_init_threads() {
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
  lockhandle = usnewlock(arenahandle);
  if (lockhandle == NULL) {
    fprintf(stderr, "lock init failed\n");
    exit(0);
  }
  usinitlock(lockhandle);
}

void spark_start_threads(int n) {
  int i;
  
  for (i=1; i<=n; i++) {
    ids[i] = i;
    (void)sproc((void (*)(void*)) smvpthread, PR_SADDR, (void *)&ids[i]);
  }
}

void spark_barrier(void) {
  barrier(barrierhandle, gip->subdomains);
}

void spark_setlock() {
  ussetlock(lockhandle);
}

void spark_unsetlock() {
  usunsetlock(lockhandle);
}
#endif



