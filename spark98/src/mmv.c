/***************************************************************************/
/* mmv.c - parallel MPI message passing SMVP kernel                        */
/*                                                                         */
/* Spark98 Kernels                                                         */
/* Copyright (C) David O'Hallaron 1998                                     */
/*                                                                         */
/* You are free to use this software without restriction. If you find that */
/* the suite is helpful to you, it would be very helpful if you sent me a  */
/* note at droh@cs.cmu.edu letting me know how you are using it.           */
/***************************************************************************/ 
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <mpi.h>

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
  int subdomains;            /* number of partition sets */ 
  int processors;            /* not used */
  int nodes;                 /* total number of local nodes on this PE */
  int mine;                  /* nodes this subdomain owns */
  int priv;                  /* unshared nodes priv <= mine <= nodes */  
  int *nodeall;              /* total nodes on each PE */
  int maxnodes;              /* max nodes on any PE */
  int elems;                 /* number of local elements */
  int matrixlen;             /* number of local nonzero block entries */
  int friends;               /* number of PEs I communicate with */
  int commlen;               /* total communicated nodes on this PE */
  int maxcommnodes;          /* max nodes communicated by any PE */

  /* 
   * global data structures 
   */
  double (*coord)[DOF];        /* geometric node coordinates */
  int (*vertex)[4];          /* nodes associated with each element */
  int *globalnode;           /* local-to-global node mapping function */
  int *globalelem;           /* local-to-global element mapping function */

  /* sparse matrix in compressed sparse row format */
  int *matrixcol;            /* K[i] is in column matrixcol[i] */
  int *matrixindex;          /* row i starts at K[matrixindex[i]] */

  /* communication schedule */
  int *comm;                 /* local node number to communicate */
  int *commindex;            /* subdomain i starts at comm[commindex[i]] */

  /* status and request vectors for communication phase */
  MPI_Status *statusvec;    
  MPI_Request *requestvec;

  /* send and receive buffers for communication phase */
  double *sendbuf;
  double *recvbuf;
  
  /* 
   * the following are determined by MPI calls 
   * there are two groups: rgroup (runtime group) and and cgroup 
   * (compute subgroup). cgroup is a subset of rgroup. We need two
   * groups because the number of subdomains might be different than
   * the number of PE's, e.g., T3D programs must run on a power of 
   * two PE's. 
   */
  
  /* runtime group */
  MPI_Group rgroup;  /* runtime group 0, ..., rsize-1 */ 
  int rsize;         /* total number of processors in runtime group */
  
  /* compute subgroup */
  MPI_Group cgroup; /* compute subgroup 0, .., asize-2 */
  MPI_Comm ccomm;   /* compute subgroup communicator */
  int csize;        /* size of compute subgroup */
  
  /* my rank and the rank of the io processor */
  int rank;         /* my rank */
  int ioproc;       /* rank of the i/o proc */
} gi, *gip=&gi;

/* timing variables */
double starttime, startctime;
double secs, csecs;

/* w1 = K1 * v1 */
double (*K1)[DOF][DOF];     /* sparse matrix */
double (*v1)[DOF];          /* dense vector */
double (*w1)[DOF];          /* dense vector */

/* w2 = K2 * v2 */
double (*K2)[DOF][DOF];     /* sparse matrix */
double (*v2)[DOF];          /* dense vector */
double (*w2)[DOF];          /* dense vector */

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
void load_global_master(struct gi *gip);
void load_global_slave(struct gi *gip);
void load_nodes_master(struct gi *gip);
void load_nodes_slave(struct gi *gip);
void load_elems_master(struct gi *gip);
void load_elems_slave(struct gi *gip);
void load_matrix_master(struct gi *gip);
void load_matrix_slave(struct gi *gip);
void load_comm_master(struct gi *gip);
void load_comm_slave(struct gi *gip);

/* routine that assembles the sparse matrix and initial vector */
void assemble_matrix(double (*K)[DOF][DOF], struct gi *gip);
void assemble_vector(double (*v)[DOF], struct gi *gip);

/* sparse matrix vector multiply routines */
void smvpthread(struct gi *gip);
void zero_vector(double (*v)[DOF], int firstrow, int numrows);
void local_smvp(int nodes, double (*A)[DOF][DOF], int *Acol, int *Aindex, 
		double (*v)[DOF], double (*w)[DOF], int firstrow, int numrows);
void full_assemble(double (*v)[DOF], struct gi *gip);

/* misc routines */
void info(struct gi *gip);
void usage(struct gi *gip);
void prglobal(struct gi *gip);
void prnodes(struct gi *gip);
void prelems(struct gi *gip);
void prmatrix(struct gi *gip);
void prcomm(struct gi *gip);
void local_printmatrix3(double (*A)[DOF][DOF], struct gi *gip);
void local_printvec3(double (*v)[DOF], int n, struct gi *gip);
void printnodevector(double (*v)[DOF], int n, struct gi *gip);

/*
 * main program 
 */
void main(int argc, char **argv) {
  int i, j;
  double mflops, global_mflops;
  double max_secs, max_csecs;
  int maxnonzeros, minnonzeros;

  init(argc, argv, gip);

  /* 
   * allocate contiguous storage for the matrix and vectors 
   */
  if (!(K1 = (void *)malloc((gip->matrixlen+1)*DOF*DOF*sizeof(double)))) {
    fprintf(stderr, "%s: couldn't malloc K1(%d)\n", 
	    gip->progname, gip->matrixlen);
    bail(gip);
  }
  if (!(K2 = (void *)malloc((gip->matrixlen+1)*DOF*DOF*sizeof(double)))) {
    fprintf(stderr, "%s: couldn't malloc K2(%d)\n", 
	    gip->progname, gip->matrixlen);
    bail(gip);
  }
  if (!(v1 = (void *)malloc(gip->nodes * DOF * sizeof(double)))) {
    fprintf(stderr, "%s: couldn't malloc v1(%d)\n", gip->progname, gip->nodes);
    bail(gip);
  }
  if (!(v2 = (void *)malloc(gip->nodes * DOF * sizeof(double)))) {
    fprintf(stderr, "%s: couldn't malloc v2(%d)\n", gip->progname, gip->nodes);
    bail(gip);
  }
  if (!(w1 = (void *)malloc(gip->nodes * DOF * sizeof(double)))) {
    fprintf(stderr, "%s: couldn't malloc w1(%d)\n", gip->progname, gip->nodes);
    bail(gip);
  }
  if (!(w2 = (void *)malloc(gip->nodes * DOF * sizeof(double)))) {
    fprintf(stderr, "%s: couldn't malloc w2(%d)\n", gip->progname, gip->nodes);
    bail(gip);
  }

  /* 
   * generate the sparse matrix coefficients 
   */
  MPI_Barrier(gip->ccomm);
  if (gip->rank == 0 && !gip->quiet) 
    fprintf(stderr, "%s: Computing sparse matrix coefficients.", 
	    gip->progname);
  assemble_matrix(K1, gip);
  assemble_matrix(K2, gip);
  assemble_vector(v1, gip);
  assemble_vector(v2, gip);

  if (!gip->quiet && gip->rank == 0) {
    fprintf(stderr, " Done.\n");
    fflush(stderr);
  }
  
  /* 
   * Do the actual SMVP operation 
   */
  if (gip->rank == 0 && !gip->quiet) {
    fprintf(stderr, "%s: Performing %d SMVP pairs (n=%d) on %d PEs.", 
	    gip->progname, gip->iters, gip->globalnodes, gip->subdomains);
  }
  smvpthread(gip);

  if (!gip->quiet && gip->rank == 0) {
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
  MPI_Barrier(gip->ccomm);
  mflops = 
    (double)((2.0*gip->matrixlen - gip->nodes)  /* nonzero blocks */
	     *DOF*DOF                           /* DOF^2 numbers/block */
	     *2.0)                              /* 2 flops/block */
	     / 1000000.0;

  /* get the total number of mflops */
  MPI_Reduce(&mflops, &global_mflops, 1, MPI_DOUBLE, MPI_SUM, 0, gip->ccomm);
  MPI_Bcast(&global_mflops, 1, MPI_DOUBLE, 0, gip->ccomm);	

  /* get the longest execution time */
  MPI_Reduce(&secs, &max_secs, 1, MPI_DOUBLE, MPI_MAX, 0, gip->ccomm);
  MPI_Bcast(&max_secs, 1, MPI_DOUBLE, 0, gip->ccomm);	
  MPI_Reduce(&csecs, &max_csecs, 1, MPI_DOUBLE, MPI_MAX, 0, gip->ccomm);
  MPI_Bcast(&max_csecs, 1, MPI_DOUBLE, 0, gip->ccomm);	

  /* get the maximum and minimum loads */
  MPI_Reduce(&gip->matrixlen, &maxnonzeros, 1, MPI_INT, MPI_MAX, 0, gip->ccomm);
  MPI_Bcast(&maxnonzeros, 1, MPI_INT, 0, gip->ccomm);	
  MPI_Reduce(&gip->matrixlen, &minnonzeros, 1, MPI_INT, MPI_MIN, 0, gip->ccomm);
  MPI_Bcast(&minnonzeros, 1, MPI_INT, 0, gip->ccomm);	
  
  if (gip->rank == 0) {
    fprintf(stderr, 
	    "%s: %s %.6lf Mf %.6lf s [%.6lf/%.6lf/%.0lf%%] %.1lf Mf/s [%d/%d/%.0lf%%]\n",
	    gip->progname, gip->packfilename, global_mflops, max_secs, 
	    max_secs - max_csecs, max_csecs, 
	    ((max_secs-max_csecs)/max_secs)*100.0, 
	    global_mflops/max_secs,
	    minnonzeros, maxnonzeros, 
	    (double)((double)minnonzeros/(double)maxnonzeros)*100.0);
  }

  /* 
   * print results if asked for
   */
  if (gip->output) {
    printnodevector(w1, gip->globalnodes, gip);
  }


  /*
   * Clean up
   */
  if (gip->rank == 0 && !gip->quiet) {
    fprintf(stderr, "%s: Done.\n", gip->progname);
  }
  finalize(gip);
}

void smvpthread(struct gi *gip) {
  int i;

  MPI_Barrier(gip->ccomm);
  csecs = 0.0;
  starttime = MPI_Wtime();
  for (i=0; i<gip->iters; i++) {

    /* w1 = K1 * v1 */
    zero_vector(w1, 0, gip->nodes);
    local_smvp(gip->nodes, K1, gip->matrixcol, gip->matrixindex, v1, w1, 0, gip->nodes);
    startctime = MPI_Wtime();
    full_assemble(w1, gip);
    csecs += (MPI_Wtime() - startctime);
    
    /* w2 = K2 * v2 */
    zero_vector(w2, 0, gip->nodes);
    local_smvp(gip->nodes, K2, gip->matrixcol, gip->matrixindex, v2, w2, 0, gip->nodes);
    startctime = MPI_Wtime();
    full_assemble(w2, gip);
    csecs += (MPI_Wtime() - startctime);
  }
  secs = (MPI_Wtime() - starttime)/(gip->iters*2.0);
  csecs = csecs / (gip->iters*2.0);
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
	      bail(gip);
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
void full_assemble(double (*v)[DOF], struct gi *gip) {
  int i, j, friend, cnt;
  
  MPI_Barrier(gip->ccomm);
  
  friend = 0;
  for (i = 0; i < gip->subdomains; i++) {
    if (gip->commindex[i] < gip->commindex[i+1]) {
      MPI_Irecv(&gip->recvbuf[gip->commindex[i]*3], 
		(gip->commindex[i+1]*3) - (gip->commindex[i]*3),
		MPI_DOUBLE, i, MPI_ANY_TAG, gip->ccomm,
		&gip->requestvec[friend]);
      friend++;
    }
  }
  
  for (i = 0; i < gip->subdomains; i++) {
    if (gip->commindex[i] < gip->commindex[i+1]) {
      cnt = 0;
      for (j = gip->commindex[i]; j < gip->commindex[i+1]; j++) {
	gip->sendbuf[cnt++] = v[gip->comm[j]][0];
	gip->sendbuf[cnt++] = v[gip->comm[j]][1];
	gip->sendbuf[cnt++] = v[gip->comm[j]][2];
      }
      MPI_Send(gip->sendbuf, cnt, MPI_DOUBLE, i, 0, gip->ccomm);
    }
  }
  
  MPI_Waitall(gip->friends, gip->requestvec, gip->statusvec);
  
  cnt = 0;
  for (i = 0; i < gip->commlen; i++) {
    v[gip->comm[i]][0] += gip->recvbuf[cnt++];
    v[gip->comm[i]][1] += gip->recvbuf[cnt++];
    v[gip->comm[i]][2] += gip->recvbuf[cnt++];
  }
  MPI_Barrier(gip->ccomm);
}

void init(int argc, char **argv, struct gi *gip) {
  char *sp;    
  int ibuf[6];
  int crange[1][3];   
  int rrank, crank;
  struct stat st;

  /* 
   * Initialize MPI.
   */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &gip->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &(gip->rsize));
  MPI_Comm_group(MPI_COMM_WORLD, &(gip->rgroup));
  
  /* 
   * There is no need to see the entire path name, the base will do. 
   */ 
  for (sp = argv[0]+strlen(argv[0]); (sp != argv[0]) && *sp != '/'; sp--)
    ;
  if (*sp == '/')
    strcpy(gip->progname, sp+1);
  else    
    strcpy(gip->progname, sp);

  parsecommandline(argc, argv, gip);

  /*
   * Do some consistency checks on process 0 and let everyone 
   * else know how many subdomains to expect.
   */
  if (gip->rank == 0) { 
    if (stat(gip->packfilename, &st) < 0) {
      fprintf(stderr, "%s: Can't find %s \n", 
	      gip->progname, gip->packfilename);
      bail(gip);
    }
    if (!(gip->packfile = fopen(gip->packfilename, "r"))) {
      fprintf(stderr, "%s: Can't open %s\n", gip->progname, gip->packfilename);
      bail(gip);
    }

    /* now make sure we have a valid packfile and that there are enough */
    /* runtime processes for the problem */
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

    if ((gip->subdomains) > gip->rsize) {
      fprintf(stderr, "%s: The mesh consists of %d subdomains. Please\n",
	      gip->progname, gip->subdomains);
      fprintf(stderr, "rerun the program using at least %d processes.\n", 
	      gip->subdomains);
      bail(gip);
    }
  }

  /* let all processes know how the number of subdomains in the mesh */
  MPI_Bcast(&gip->subdomains, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* 
   * Create the compute subgroup (one process per subdomain) 
   */
  crange[0][0] = 0;
  crange[0][1] = gip->subdomains-1;
  crange[0][2] = 1;
  MPI_Group_range_incl(gip->rgroup, 1, crange, &(gip->cgroup)); 
  MPI_Group_rank(gip->cgroup, &crank);
  MPI_Group_size(gip->cgroup, &(gip->csize));
  MPI_Comm_create(MPI_COMM_WORLD, gip->cgroup, &(gip->ccomm));
  gip->ioproc = gip->csize-1;
  
  /* 
   * Run as a compute process 
   */
  if (gip->rank < gip->subdomains) {
    if ((gip->rank != crank)) {
      fprintf(stderr, "%s: inconsistent ranks on compute process\n", 
	      gip->progname);
      bail(gip);
    }	    

    /* input the packfile */
    load_pack(gip);

    /* 
     * preallocate some MPI status and request buffers  that are
     * needed for the asynchronous communication phase 
     */
    if (!(gip->statusvec = (MPI_Status *) 
	  malloc(gip->subdomains * sizeof(MPI_Status)))) {
      fprintf(stderr, "%s: couldn't allocate statusvec\n", gip->progname);
      bail(gip);
    }
    
    if (!(gip->requestvec = (MPI_Request *) 
	  malloc(gip->subdomains * sizeof(MPI_Request)))) {
      fprintf(stderr, "%s: couldn't allocate request\n", gip->progname);
      bail(gip);
    }
    
    /*
     * preallocate send and receive buffers for the communication phase 
     */
    if (gip->maxcommnodes > 0) {
      if (!(gip->sendbuf = (double *)
	    malloc(gip->maxcommnodes * 3 * sizeof(double)))) {
	fprintf(stderr, "%s: couldn't allocate sendbuf\n", gip->progname);
	bail(gip);
      }
      
      if (!(gip->recvbuf = (double *) 
	    malloc(gip->maxcommnodes * 3 * sizeof(double)))) {
	fprintf(stderr, "%s: couldn't allocate recvbuf\n", gip->progname);
	bail(gip);
      }
    }
  }
    
  /* 
   * Run as a dummy process, wait for compute processes, 
   * then terminate. 
   */
  else {
    finalize(gip);
  }
}

/*
 * load_pack - load and distribute the packfile
 */
void load_pack(struct gi *gip) {
  struct stat st;
  
  if (stat(gip->packfilename, &st) < 0) {
    fprintf(stderr, "%s: Can't find %s \n", 
	    gip->progname, gip->packfilename);
    bail(gip);
  }
  
  if (!(gip->packfile = fopen(gip->packfilename, "r"))) {
    fprintf(stderr, "%s: Can't open %s\n", 
	    gip->progname, gip->packfilename);
    bail(gip);
  }
  
  if (gip->rank == gip->ioproc) {
    load_global_master(gip);
    load_nodes_master(gip);
    load_elems_master(gip);
    load_matrix_master(gip);
    load_comm_master(gip);
  }
  else {
    load_global_slave(gip);
    load_nodes_slave(gip);
    load_elems_slave(gip);
    load_matrix_slave(gip);
    load_comm_slave(gip);
  }
}

void load_global_master(struct gi *gip) {
  int ibuf[6];
  
  fscanf(gip->packfile, "%d %d\n", &gip->globalnodes, &gip->mesh_dim);
  fscanf(gip->packfile, "%d %d\n", &gip->globalelems, &gip->corners);
  fscanf(gip->packfile, "%d\n", &gip->subdomains);
  fscanf(gip->packfile, "%d\n", &gip->processors);
  ibuf[0] = gip->globalnodes;
  ibuf[1] = gip->mesh_dim;
  ibuf[2] = gip->globalelems;
  ibuf[3] = gip->corners;
  ibuf[4] = gip->subdomains;
  ibuf[5] = gip->processors;

  if ((gip->subdomains < 1) || (gip->subdomains > 1024) ||
      ((gip->mesh_dim != 2) && (gip->mesh_dim != 3)) ||
      ((gip->corners != 3) && (gip->corners != 4)) ||
      gip->processors != gip->subdomains) {
    fprintf(stderr, "%s: the input file doesn't appear to be a packfile\n",
	    gip->progname);
    bail(gip);
  }

  MPI_Bcast(ibuf, 6, MPI_INT, gip->ioproc, gip->ccomm);
  
#ifdef DEBUGGLOBAL    
  prglobal(gip);
#endif
}

void load_global_slave(struct gi *gip) {
  int ibuf[6];
  
  MPI_Bcast(ibuf, 6, MPI_INT, gip->ioproc, gip->ccomm);
  gip->globalnodes = ibuf[0];
  gip->mesh_dim = ibuf[1];
  gip->globalelems = ibuf[2];
  gip->corners = ibuf[3];
  gip->subdomains = ibuf[4];
  gip->processors = ibuf[5];
  
#ifdef DEBUGGLOBAL    
  prglobal(gip);
#endif
}

void load_nodes_master(struct gi *gip) {
  int i, j, k;
  int *nodepriv;
  int *nodemine;
  double *fbuf, *fbufp;
  int *buf, *bufp;
  int wordint;
  double wordfloat;

  MPI_Barrier(gip->ccomm);

  if (!gip->quiet) {
    fprintf(stderr, "%s: Reading nodes", gip->progname);
    fflush(stderr);
  }
  if (!(gip->nodeall = (int *) malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate nodeall\n", 
	    gip->progname);
    bail(gip);
  }
  if (!(nodepriv = (int *) malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate nodepriv\n", gip->progname);
    bail(gip);
  }
  if (!(nodemine = (int *) malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate nodemine\n", gip->progname);
    bail(gip);
  }

  gip->maxnodes = 0;
  for (i = 0; i < gip->subdomains; i++) {
    fscanf(gip->packfile, "%d %d %d\n", 
	   &gip->nodeall[i], &nodemine[i], &nodepriv[i]);
    if (gip->nodeall[i] > gip->maxnodes)
      gip->maxnodes = gip->nodeall[i];
  }

  MPI_Bcast(&gip->maxnodes, 1, MPI_INT, gip->ioproc, gip->ccomm);
  MPI_Bcast(gip->nodeall, gip->subdomains, MPI_INT, gip->ioproc, gip->ccomm);
    
  if (!(fbuf = (double *)malloc(gip->maxnodes*gip->mesh_dim*sizeof(double)))) {
    fprintf(stderr, "%s: couldn't allocate local node buffer\n", 
	    gip->progname);
    bail(gip);
  }

  if (!(buf = (int *) malloc(gip->maxnodes * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't alloc global node index buffer\n", 
	    gip->progname);
    bail(gip);
  }
    
  for (i = 0; i < gip->subdomains; i++) {
    if (!gip->quiet) {
      fprintf(stderr, ".");
      fflush(stderr);
    }

    /* node header */
    bufp = buf;
    *bufp++ = gip->nodeall[i];
    *bufp++ = nodemine[i];
    *bufp++ = nodepriv[i];
    MPI_Send(buf, (int)(bufp-buf), MPI_INT, i, 0, gip->ccomm);

    /* node data */
    bufp = buf;
    fbufp = fbuf;
    for (j = 0; j < gip->nodeall[i]; j++) {

      /* global node number */
      fscanf(gip->packfile, "%d", &wordint);
      *bufp++ = wordint;

      /* node data */
      for (k = 0; k < gip->mesh_dim; k++) {
	fscanf(gip->packfile, "%lf", &wordfloat);
	if ((fbufp-fbuf) > gip->maxnodes * gip->mesh_dim) {
	  fprintf(stderr, "%s: node buffers too small (%d > %d)\n", 
		  gip->progname, (int)(fbufp-fbuf),
		  gip->maxnodes * gip->mesh_dim);
	  bail(gip);
	}
	*fbufp++ = wordfloat;
      }
    }
    if (i < gip->ioproc) {
      MPI_Send(buf, (int)(bufp-buf), MPI_INT, i, 0, gip->ccomm);
      MPI_Send(fbuf, (int)(fbufp-fbuf), MPI_DOUBLE, i, 0, gip->ccomm);
    }
    else {
      gip->nodes = gip->nodeall[i];
      gip->priv = nodepriv[i];
      gip->mine = nodemine[i];
      gip->globalnode = (int *)buf;
      gip->coord = (void *)fbuf;
    }
  }

  (void)free(nodepriv);
  (void)free(nodemine);

  if (!gip->quiet) {
    fprintf(stderr, " Done.\n");
    fflush(stderr);
  }

#ifdef DEBUGNODES
  prnodes(gip);
#endif
}

void load_nodes_slave(struct gi *gip) {
  int ibuf[3];
  MPI_Status status;
  
  MPI_Barrier(gip->ccomm);

  MPI_Bcast(&gip->maxnodes, 1, MPI_INT, gip->ioproc, gip->ccomm);

  if (!(gip->nodeall = (int *) malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate nodeall\n", 
	    gip->progname);
    bail(gip);
  }
  MPI_Bcast(gip->nodeall, gip->subdomains, MPI_INT, gip->ioproc, gip->ccomm);
  MPI_Recv(ibuf, 3, MPI_INT, 
	   gip->ioproc, MPI_ANY_TAG, gip->ccomm, &status);
  
  gip->nodes = ibuf[0];
  gip->mine = ibuf[1]; 
  gip->priv = ibuf[2]; 
  
  if (!(gip->coord = (void *) 
	malloc(gip->nodes * gip->mesh_dim * sizeof(double)))) {
    fprintf(stderr, "%s: couldn't allocate coord(%d)\n", 
	    gip->progname, gip->nodes);
    bail(gip);
  }
  
  if (!(gip->globalnode = (int *) malloc(gip->nodes * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't alloc globalnode\n", 
	    gip->progname);
    bail(gip);
  }
  
  MPI_Recv(gip->globalnode, gip->nodes, MPI_INT, 
	   gip->ioproc, MPI_ANY_TAG, gip->ccomm, &status);

  MPI_Recv(gip->coord, gip->nodes * gip->mesh_dim, MPI_DOUBLE, 
	   gip->ioproc, MPI_ANY_TAG, gip->ccomm, &status);
  

#ifdef DEBUGNODES
  prnodes(gip);
#endif 
}

void load_elems_master(struct gi *gip) {
  int i, j, k;
  int wordint;
  int *elemcnt;
  int maxelems;
  int *buf, *bufp;
  int *buf2, *buf2p;

  MPI_Barrier(gip->ccomm);

  if (!gip->quiet) {
    fprintf(stderr, "%s: Reading elements", gip->progname);
    fflush(stderr);
  }
  if (!(elemcnt = (int *) malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate elemcnt\n", gip->progname);
    bail(gip);
  }

  maxelems = 0;
  for (i = 0; i < gip->subdomains; i++) {
    fscanf(gip->packfile, "%d\n", &elemcnt[i]);
    if (elemcnt[i] > maxelems)
      maxelems = elemcnt[i];
  }
    
  if (!(buf = (int *) malloc(maxelems * gip->corners * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate element corner buf\n", 
	    gip->progname);
    bail(gip);
  }

  if (!(buf2 = (int *) malloc(maxelems * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't alloc global element buf\n", 
	    gip->progname);
    bail(gip);
  }

  for (i = 0; i < gip->subdomains; i++) {

    if (!gip->quiet) {
      fprintf(stderr, ".");
      fflush(stderr);
    }

    /* element header */
    bufp = buf;
    *bufp++ = elemcnt[i];
    MPI_Send(buf, 1, MPI_INT, i, 0, gip->ccomm);
	
    /* element data */
    bufp = buf;
    buf2p = buf2;
    for (j = 0; j < elemcnt[i]; j++) {

      /* global elem number */
      fscanf(gip->packfile, "%d", &wordint);
      *buf2p++ = wordint;
    
      /* elem data */
      for (k = 0; k < gip->corners; k++) {
	fscanf(gip->packfile, "%d", &wordint);
	if ((bufp-buf) > maxelems*gip->corners) {
	  fprintf(stderr, "%s: elem buffers too small\n", 
		  gip->progname);
	  bail(gip);
	}
	*bufp++ = wordint;
      }
    }

    if (i < gip->ioproc) {
      MPI_Send(buf2, (int)(buf2p-buf2), MPI_INT, i, 0, gip->ccomm);
      MPI_Send(buf, (int)(bufp-buf), MPI_INT, i, 0, gip->ccomm);
    }
    else {
      gip->elems = elemcnt[i];
      gip->globalelem = (int *)buf2;
      gip->vertex = (void *)buf;
    }
  }

  (void)free(elemcnt);

  if (!gip->quiet) {
    fprintf(stderr, " Done.\n");
    fflush(stderr);
  }

#ifdef DEBUGELEMS
  prelems(gip);
#endif
}  

void load_elems_slave(struct gi *gip) {
  MPI_Status status;
  
  MPI_Barrier(gip->ccomm);

  MPI_Recv(&gip->elems, 1, MPI_INT, 
	   gip->ioproc, MPI_ANY_TAG, gip->ccomm, &status);

  if (!(gip->vertex = (void *) malloc(gip->elems*gip->corners*sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate vertex\n", gip->progname);
    bail(gip);
  }

  if (!(gip->globalelem = (void *) malloc(gip->elems*sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate globalelem\n", gip->progname);
    bail(gip);
  }

  MPI_Recv(gip->globalelem, gip->elems, MPI_INT, 
	   gip->ioproc, MPI_ANY_TAG, gip->ccomm, &status);
  
  MPI_Recv(gip->vertex, gip->elems*gip->corners, MPI_INT, 
	   gip->ioproc, MPI_ANY_TAG, gip->ccomm, &status); 
  
#ifdef DEBUGELEMS
  prelems(gip);
#endif
}

void load_matrix_master(struct gi *gip) {
  int i, k, loop1;
  int oldrow, newrow;
  int maxcsrwords;
  int *csrwords;
  int *buf;
  int *idxbuf;

  if (!gip->quiet) {
    fprintf(stderr, "%s: Reading matrix", gip->progname);
    fflush(stderr);
  }

  if (!(csrwords = (int *) malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate csrwords\n", gip->progname);
    bail(gip);
  }

  /* get the number of i,j pairs on each subdomain */
  maxcsrwords = 0;
  for (i = 0; i < gip->subdomains; i++) { 
    fscanf(gip->packfile, "%d %*d\n", &csrwords[i]); 
    if (csrwords[i] > maxcsrwords)
      maxcsrwords = csrwords[i];
  }

  if (!(buf = (int *) malloc((maxcsrwords+1) * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate col buffer\n", gip->progname);
    bail(gip);
  }

  if (!(idxbuf = (int *) malloc((gip->maxnodes+1) * sizeof(int)))){
    fprintf(stderr, "%s: couldn't allocate index buffer\n", gip->progname);
    bail(gip);
  }
    
  /* build the CSR format for each subdomain and send it */
  MPI_Barrier(gip->ccomm);
  for (i = 0; i < gip->subdomains; i++) {

    if (!gip->quiet) {
      fprintf(stderr, ".");
      fflush(stderr);
    }

    MPI_Send(&csrwords[i], 1, MPI_INT, i, 0, gip->ccomm);
    oldrow = -1; 
    for (loop1 = 0; loop1 < csrwords[i]; loop1++) {

      fscanf(gip->packfile, "%d", &newrow);
      fscanf(gip->packfile, "%d", &buf[loop1]);
      while (oldrow < newrow) {
	if (oldrow+1 >= maxcsrwords) {
	  fprintf(stderr, 
		  "%s: index buffer(1) too small (%d >= %d)\n", 
		  gip->progname, oldrow+1, maxcsrwords);
	  bail(gip);
	}
	idxbuf[++oldrow] = loop1;
      }
    }
    while (oldrow < gip->nodeall[i]) {
      if (oldrow+1 >= maxcsrwords) {
	fprintf(stderr, "%s: index buffer(2) too small (%d >= %d)\n", 
		gip->progname, oldrow+1, maxcsrwords);
	bail(gip);
      }
      idxbuf[++oldrow] = csrwords[i];
    }
    if (i < gip->ioproc) {
      MPI_Send(buf, csrwords[i], MPI_INT, i, 0, gip->ccomm);
      MPI_Send(idxbuf, gip->nodeall[i]+1, MPI_INT, i, 0, gip->ccomm);
    } 
    else {
      gip->matrixlen = csrwords[i];
      gip->matrixcol = (void *)buf;
      gip->matrixindex = (void *)idxbuf;
    }
  }   

  (void)free(csrwords);

  if (!gip->quiet) {
    fprintf(stderr, " Done.\n");
    fflush(stderr);
  }

#ifdef DEBUGMATRIX
  prmatrix(gip);
#endif
}

void load_matrix_slave(struct gi *gip) {
  MPI_Status status;

  MPI_Barrier(gip->ccomm);

  MPI_Recv(&gip->matrixlen, 1, MPI_INT, 
	   gip->ioproc, MPI_ANY_TAG, gip->ccomm, &status);

  if (!(gip->matrixcol = (int *) malloc(gip->matrixlen * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate matrixcol\n", gip->progname);
    bail(gip);
  }

  if (!(gip->matrixindex = (int *) malloc((gip->nodes+1) * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate matrixindex\n", gip->progname);
    bail(gip);
  }
    
  MPI_Recv(gip->matrixcol, gip->matrixlen, MPI_INT, 
	   gip->ioproc, MPI_ANY_TAG, gip->ccomm, &status);

  MPI_Recv(gip->matrixindex, gip->nodes+1, MPI_INT, 
	   gip->ioproc, MPI_ANY_TAG, gip->ccomm, &status);


#ifdef DEBUGMATRIX
  prmatrix(gip);
#endif
}

void load_comm_master(struct gi *gip) {
  int i, j, k;
  int count;
  int *commnodes; /*subdomain i shares a total of commnodes[i] nodes */
  int *buf;
  
  MPI_Barrier(gip->ccomm);

  if (!gip->quiet) {
    fprintf(stderr, "%s: Reading communication info", gip->progname);
    fflush(stderr);
  }
  
  if (!(commnodes = (int *) malloc(gip->subdomains * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate commnodes\n", gip->progname);
    bail(gip);
  }
  
  gip->maxcommnodes = 0;
  for (i = 0; i < gip->subdomains; i++) {
    fscanf(gip->packfile, "%d\n", &commnodes[i]);
    if (commnodes[i] > gip->maxcommnodes) {
      gip->maxcommnodes = commnodes[i];
    }
  }
  
  MPI_Bcast(&gip->maxcommnodes, 1, MPI_INT, gip->ioproc, gip->ccomm);

  if (!(gip->comm = (int *) malloc((gip->maxcommnodes+1) * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate comm\n", gip->progname);
    bail(gip);
  }
  
  if (!(gip->commindex = (int *) malloc((gip->subdomains+1) * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate commindex\n", gip->progname);
    bail(gip);
  }
  

  for (i = 0; i < gip->subdomains; i++) {
    
    if (!gip->quiet) {
      fprintf(stderr, ".");
      fflush(stderr);
    }
    
    gip->commlen = 0;
    gip->friends = 0;
    for (j = 0; j < gip->subdomains; j++) {
      
      gip->commindex[j] = gip->commlen;

      /* subdomain i shares count nodes with subdomain j */
      fscanf(gip->packfile, "%d", &count);
      if (count > gip->maxcommnodes) {
	fprintf(stderr, "%s: count exceeds maxcommnodes\n", gip->progname);
	bail(gip);
      }
      
      if (count > 0)
	gip->friends++;

      for (k = 0; k < count; k++) {
	fscanf(gip->packfile, "%d", &gip->comm[gip->commlen++]);
      }
    }
    gip->commindex[gip->subdomains] = gip->commlen;

    if (gip->commlen != commnodes[i]) {
      fprintf(stderr, "%s: inconsistent comm lengths\n", gip->progname);
      bail(gip);
    }

    if (i < gip->ioproc) {
	MPI_Send(&gip->commlen, 1, MPI_INT, i, 0, gip->ccomm);
	MPI_Send(&gip->friends, 1, MPI_INT, i, 0, gip->ccomm);
	MPI_Send(gip->comm, gip->commlen, MPI_INT, i, 0, gip->ccomm);
	MPI_Send(gip->commindex, gip->subdomains+1, MPI_INT, i, 0, gip->ccomm);
    }
  }

  (void)free(commnodes);
  
  if (!gip->quiet) {
    fprintf(stderr, " Done.\n");
    fflush(stderr);
  }

#ifdef DEBUGCOMM
    prcomm(gip);
#endif
}    

void load_comm_slave(struct gi *gip) {
  MPI_Status status;
  
  MPI_Barrier(gip->ccomm);
  MPI_Bcast(&gip->maxcommnodes, 1, MPI_INT, gip->ioproc, gip->ccomm);
  MPI_Recv(&gip->commlen, 1, MPI_INT, 
	   gip->ioproc, MPI_ANY_TAG, gip->ccomm, &status);
  MPI_Recv(&gip->friends, 1, MPI_INT, 
	   gip->ioproc, MPI_ANY_TAG, gip->ccomm, &status);

  if (!(gip->comm = (int *) malloc((gip->commlen+1) * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate comm\n", gip->progname);
    bail(gip);
  }

  if (!(gip->commindex = (int *) malloc((gip->subdomains+1) * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate commindex\n", gip->progname);
    bail(gip);
  }

  MPI_Recv(gip->comm, gip->commlen, MPI_INT, 
	   gip->ioproc, MPI_ANY_TAG, gip->ccomm, &status);
  MPI_Recv(gip->commindex, gip->subdomains+1, MPI_INT, 
	   gip->ioproc, MPI_ANY_TAG, gip->ccomm, &status);

#ifdef DEBUGCOMM
    prcomm(gip);
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
    bail(gip);
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
  if (gip->rank == 0) {
    printf("\n");
    printf("You are running the %s kernel from the Spark98 Kernels.\n", gip->progname);
    printf("Copyright (C) 1998, David O'Hallaron, Carnegie Mellon University.\n");
    printf("You are free to use this software without restriction. If you find that");
    printf("the suite is helpful to you, it would be very helpful if you sent me a ");
    printf("note at droh@cs.cmu.edu letting me know how you are using it.          "); 
    printf("\n");
    printf("%s is a parallel message passing program. Each processor performs\n",gip->progname);
    printf("its own local SMVP operation using partially assembled local matrices\n");
    printf("and vectors, followed by an assembly phase that combines the partially\n");
    printf("assembled output vectors.\n");
    printf("\n");
    printf("%s [-hOQ] [-i<n>] packfilename\n\n", gip->progname);
    printf("Command line options:\n\n");
    printf("    -h    Print this message and exit.\n");
    printf("    -i<n> Do n iterations of the SMVP pairs (default %d).\n", 
	   DEFAULT_ITERS);
    printf("    -O    Print the output vector to stdout.\n");
    printf("    -Q    Quietly suppress all explanations.\n");
    printf("\n");
    printf
      ("Input packfiles are produced using the Archimedes tool chain.\n");
    printf("\n");
    printf("Example: mpirun -np 8 %s -O -i10 sf5.8.pack\n", gip->progname);
  }
}

void usage(struct gi *gip) {
  if (gip->rank == 0) {
    printf("\n");
    printf("usage: %s [-hOQ] [-i<n>] packfilename\n\n", gip->progname);
  }
}


/*
 * printnodevector - print one column of a global nodevector
 */
void printnodevector(double (*v)[DOF], int n, struct gi *gip) {
  MPI_Status status;
  int i, j, nitems;
  int cmd = SEND_CMD;
  double *gvec, *fvec;
  int *ivec;


  MPI_Barrier(gip->ccomm);

  /* allocate scratch buffers for print operations */
  if (!(fvec = (double *)malloc((gip->maxnodes) * sizeof(double)))) {
    fprintf(stderr, "%s: couldn't allocate fvec\n", gip->progname);
    bail(gip);
  }


  if (!(gvec = (double *) malloc((gip->globalnodes+1) * sizeof(double)))) {
    fprintf(stderr, "%s: couldn't allocate gvec\n", gip->progname);
    bail(gip);
  }

  if (!(ivec = (int *) malloc((gip->maxnodes) * sizeof(int)))) {
    fprintf(stderr, "%s: couldn't allocate ivec\n", gip->progname);
    bail(gip);
  }

  /* i/o master */
  if (gip->rank == 0) {
    for (j=1; j<=gip->globalnodes; j++) {
      gvec[j] = INVALID;
    }

    for (j=0; j<gip->mine; j++) {
      gvec[gip->globalnode[j]] = v[j][0];
    }
    for (i=1; i<gip->subdomains; i++) { 
      MPI_Send(&cmd, 1, MPI_INT, i, 0, gip->ccomm);
      MPI_Probe(i, MPI_ANY_TAG, gip->ccomm, &status);

      MPI_Get_count(&status, MPI_DOUBLE, &nitems);
      MPI_Recv(fvec, nitems, MPI_DOUBLE, i, MPI_ANY_TAG, gip->ccomm, &status);
      MPI_Recv(ivec, nitems, MPI_INT, i, MPI_ANY_TAG, gip->ccomm, &status);
      for (j=0; j<nitems; j++) {
	if (gvec[ivec[j]] != INVALID) {
	  fprintf(stderr, 
		  "warning: received duplicate entry in gvec[%d] from %d\n", 
		  ivec[j], i);
	}
	gvec[ivec[j]] = fvec[j];
      }
    }
    for (j=1; j<=n; j++) {
      if (gvec[j] == INVALID) {
	fprintf(stderr, "error: uninitialized value in gvec[%d]\n", j);
	bail(gip);
      }
      printf("%d %.0f\n", j, gvec[j]); 
    }
  }

  /* i/o slave */
  else {
    for (j=0; j<gip->mine; j++) {
      fvec[j] = v[j][0];
    }
    MPI_Send(fvec, gip->mine, MPI_DOUBLE, 0, 0, gip->ccomm);
    MPI_Send(gip->globalnode, gip->mine, MPI_INT, 0, 0, gip->ccomm);
  }

  free(gvec);
  free(fvec);
  free(ivec);
}

/*
 * local output routines for debugging
 */

/* emit local matrix on each subdomain */
void local_printmatrix3(double (*A)[DOF][DOF], struct gi *gip) {
  int i, j, k, l, row, col, subdomain;

  for (subdomain = 0; subdomain < gip->subdomains; subdomain++) {
    MPI_Barrier(gip->ccomm);
    if (subdomain == gip->rank) {
       for (i = 0; i < gip->nodes; i++) { 
	for (j = 0; j < 3; j++) {
	  for (k = gip->matrixindex[i]; 
	       k < gip->matrixindex[i+1];
	       k++) {
	    for (l = 0; l < 3; l++) {
	      row =  i*3 + j + 1; /*row*/
	      col =  gip->matrixcol[k]*3 
		+ l + 1; /*col*/
	      printf("%d: %d %d %.0f\n", gip->rank, row, col, A[k][j][l]);
	      printf("%d: %d %d %.0f\n", gip->rank, col, row, A[k][j][l]);
	    }
	  }
	}
      }
    }
  }
}

/* emit local vector on each subdomain */
void local_printvec3(double (*v)[DOF], int n, struct gi *gip) {
  int i, j;
  
  for (i=0; i < gip->subdomains; i++) {
    MPI_Barrier(gip->ccomm);
    if (i == gip->rank) {
      for (j = 0; j < n; j++) {
	printf("[%d]: %d (%d): %.0f\n", 
	       gip->rank, j, gip->globalnode[j], v[j][0]);
      }
      fflush(stdout);
    }
  }
}

/*
 * debug routines that print out various local data structures one
 * subdomain at a time during the initialization phase.
 */
void prglobal(struct gi *gip)
{
  int i;

  for (i = 0; i < gip->csize; i++) {
    MPI_Barrier(gip->ccomm);
    if (i == gip->rank) {
      printf("[%d]: gnodes=%d dim=%d gelem=%d corners=%d subdomains=%d procs=%d\n",
	     gip->rank, gip->globalnodes, gip->mesh_dim, 
	     gip->globalelems, gip->corners, gip->subdomains, 
	     gip->processors);
    }
    fflush(stdout);
  }
}

void prnodes(struct gi *gip) {
  int i, j, k; 
  
  for (i = 0; i < gip->subdomains; i++) {
    MPI_Barrier(gip->ccomm);
    if (i == gip->rank) {
      printf("[%d]: gip->nodes=%d gip->priv=%d gip->mine=%d\n", 
	     gip->rank, gip->nodes, gip->priv, gip->mine);
      for (j = 0; j < gip->nodes; j++) {
	printf("[%d] %d (%d): ", gip->rank, j, gip->globalnode[j]);
	for (k = 0; k < gip->mesh_dim; k++) {
	  printf("%f ", gip->coord[j][k]);
	}
	printf("\n");	
      }
      fflush(stdout);
    }
  }
}

void prelems(struct gi *gip)
{
  int i, j, k;
    
  for (i = 0; i < gip->csize; i++) {
    MPI_Barrier(gip->ccomm);
    if (i == gip->rank) {
      printf("[%d]: gip->elems=%d\n", gip->rank, gip->elems);  
      for (j = 0; j < gip->elems; j++) {
	printf("\n");    
	printf("[%d]: %d (%d): ", gip->rank, j, gip->globalelem[j]);
	for (k = 0; k < gip->corners; k++) {
	  printf("%d ", gip->vertex[j][k]);
	}
      }
      printf("\n");
      fflush(stdout);
    }
  }
}

void prmatrix(struct gi *gip)
{
  int i, j;
    
  for (i = 0; i < gip->csize; i++) {
    MPI_Barrier(gip->ccomm);
    if (gip->rank == i) {
      fflush(stdout);
      printf("[%d]: gip->matrixcol:\n", gip->rank);	
      for (j = 0; j < gip->matrixlen; j++)
	printf("[%d]: %d: %d\n", gip->rank, j, gip->matrixcol[j]); 
      printf("[%d]: gip->matrixindex:\n", gip->rank);	
      for (j = 0; j <= gip->nodes; j++)
	printf("[%d]: %d: %d\n", gip->rank, j, gip->matrixindex[j]);
      fflush(stdout);
    }
  }
}

void prcomm(struct gi *gip) {
  int i, j, k;
  
  for (i = 0; i < gip->subdomains; i++) {
    MPI_Barrier(gip->ccomm);
    if (i == gip->rank) {
      printf("[%d]: comm info (%d shared nodes, %d friends)\n", 
	     gip->rank, gip->commlen, gip->friends);
      for (j = 0; j < gip->subdomains; j++) {
	printf("[%d]: %d nodes shared with subdomain %d:\n", 
	       gip->rank, gip->commindex[j+1]-gip->commindex[j],j);
	fflush(stdout);
	for (k = gip->commindex[j]; k < gip->commindex[j+1]; k++) {
	  printf("[%d]: %d (%d)\n", gip->rank, 
		 gip->comm[k], gip->globalnode[gip->comm[k]]);
	  fflush(stdout);
	}
      }
      fflush(stdout);
    }
  }
}


/*
 * termination routines 
 */

/* orderly exit */
void finalize(struct gi *gip) 
{
  MPI_Barrier(MPI_COMM_WORLD);
  if ((gip->rank == 0) && !gip->quiet) {
    fprintf(stderr, "%s: Terminating normally.\n", gip->progname);
    fflush(stderr);
  }
  MPI_Finalize();
  exit(0);
}

/* emergency exit */
void bail(struct gi *gip)
{
  fprintf(stderr, "\n");
  fprintf(stderr, "%s: Something bad happened. Terminating abnormally.\n", 
	  gip->progname);
  fprintf(stderr, "\n");
  fflush(stderr);
  fflush(stdout);
  MPI_Abort(MPI_COMM_WORLD, 0);
}






