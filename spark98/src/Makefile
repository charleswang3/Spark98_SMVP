#
# AlphaServer 8400 using Pthreads
#
CC = cc
OPTIONS = -O2

all: smv lmv rmv hmv psmv

smv: mv.c
	$(CC) $(OPTIONS) -o smv mv.c

lmv: mv.c
	$(CC) $(OPTIONS) -DPTHREAD_LOCK -o lmv mv.c -lpthread

rmv: mv.c
	$(CC) $(OPTIONS) -DPTHREAD_REDUCE -o rmv mv.c -lpthread

psmv: mv.c
	$(CC) $(OPTIONS) -DSCHEDULE -o psmv mv.c -lpthread

hmv: hmv.c
	$(CC) $(OPTIONS) -DPTHREAD -o hmv hmv.c -pthread


clean:
	-rm -f smv lmv rmv hmv *~
