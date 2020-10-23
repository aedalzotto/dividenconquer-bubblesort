#include <mpi.h>

/* Define DEBUG by compiler's command line (-DDEBUG=1) */
#ifndef DEBUG
	#define DEBUG 1
#endif

unsigned last_pow_2(unsigned x);

int main(int argc, char *argv[])
{
	/* Initialize MPI */
	MPI_Init(&argc, &argv);

	/* Get MPI processor identification */
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* Get MPI processor number */
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(size % 2 == 0){
		if(rank == 0)
			printf("ERROR: Please use an odd number of process\n");

		MPI_Finalize();
		exit(1);
	}

#if DEBUG == 1
	/* DEBUG array with 40 entries */
	const unsigned ROOT_LEN = 40;
#else
	/* Default with 1.000.000 entries */
	/* Or configure with compiler's command line (-DN=N) */
	#ifndef N
		const unsigned ROOT_LEN = 1000000;
	#else
		const unsigned ROOT_LEN = N;
	#endif
#endif

	/* Make the rank even */
	unsigned evenrank = rank % 2 ? rank + 1 : rank;

	/* Number of leaves in the tree depth */
	unsigned depth_len = evenrank ? last_pow_2(evenrank) : 1;

	/* Find the size of the array that will be computed in this node */
	unsigned leaf_len = ROOT_LEN / depth_len;

	/* Find the parent of the leaf. This will be invalidated for root (node 0) */
	int parent = evenrank / 2 - 1;

	/* Find first child of the node */
	int child = (rank * 2) + 1;

	int *values = malloc(leaf_len*sizeof(int));

	MPI_Status status;

	if(rank == 0){
		/* Root: initial producer */
		/* Populate array in decreasing order to execute worst case */
		for(int i = 0; i < leaf_len; i++)
			values[i] = leaf_len - i;
	} else {
		MPI_Recv(values, leaf_len, MPI_INT, parent, 0, MPI_COMM_WORLD, &status);
	}

	if(child + 1 < size){
		/* Divide to childs */
		MPI_Send(values, leaf_len / 2, MPI_INT, child, 0, MPI_COMM_WORLD);
		MPI_Send(&values[leaf_len / 2], leaf_len / 2, MPI_INT, child + 1, 0, MPI_COMM_WORLD);

		MPI_Recv(values, leaf_len / 2, MPI_INT, child, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&values[leaf_len / 2], leaf_len / 2, MPI_INT, child + 1, 0, MPI_COMM_WORLD, &status);

		/* Interleave the messages */
		//interleave(a,b);
	} else {
		/* Conquer! */
		//bubblesort
	}

	if(rank == 0){
		/* Root node achieved. Show the array */
		//print_array
	} else {
		/* Send back array to parent */
		MPI_Send(values, leaf_len, MPI_INT, parent, 0, MPI_COMM_WORLD);
	}
	
	MPI_Finalize();
}

unsigned last_pow_2(unsigned x)
{
	x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
	x |= (x >> 32); /* Support up to 64 bit */
    return x - (x >> 1);
}
