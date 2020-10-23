#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/* Define DEBUG by compiler's command line (-DDEBUG=1) */
#ifndef DEBUG
	#define DEBUG 1
#endif

unsigned last_pow_2(unsigned x);
void bubblesort(int *array, const int SIZE);
void interleave(int *src_a, int *src_b, int *dest, int length);
#if DEBUG == 1
void print_array(int *array, int len);
#endif

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

	if(ROOT_LEN % 2 != 0){
		if(rank == 0)
			printf("ERROR: Please use an even number of array elements\n");

		MPI_Finalize();
		exit(1);
	}

	/* Make the rank even */
	unsigned evenrank = rank % 2 ? rank + 1 : rank;

	/* Number of leaves in the tree depth */
	unsigned depth_len = evenrank ? last_pow_2(evenrank) : 1;

	/* Find the size of the array that will be computed in this node */
	unsigned leaf_len = ROOT_LEN / depth_len;

	/* Find the parent of the leaf. This will be invalidated for root (node 0) */
	int parent = evenrank / 2 - 1;

	/* Find first child of the node. Will be invalidated if child >= size */
	int child = (rank * 2) + 1;

	int *values = malloc(leaf_len*sizeof(int));
	if(values == NULL){
		printf("P%d: Not enough memory to allocate main array of length %d\n", rank, leaf_len);
		MPI_Finalize();
		exit(1);
	}

	MPI_Status status;

#if DEBUG == 0
	double then = 0;
#endif
	if(rank == 0){
		/* Root: initial producer */
		/* Populate array in decreasing order to execute worst case */
	#if DEBUG == 1
		printf("P%d: Populating main array\n", rank);
	#endif
		for(int i = 0; i < leaf_len; i++)
			values[i] = leaf_len - i;

	printf("P%d: Sorting array of %d elements in %d process\n", rank, ROOT_LEN, size);
	#if DEBUG == 1
		print_array(values, leaf_len);
	#else
		then = MPI_Wtime();
	#endif
	} else {
	#if DEBUG == 1
		printf("P%d: Receiving array from parent %d\n", rank, parent);
	#endif
		MPI_Recv(values, leaf_len, MPI_INT, parent, 0, MPI_COMM_WORLD, &status);
	}

	if(child + 1 < size){
		/* Divide to childs */
	#if DEBUG == 1
		printf("P%d: Sending array to childs %d and %d\n", rank, child, child + 1);
	#endif
		MPI_Send(values, leaf_len / 2, MPI_INT, child, 0, MPI_COMM_WORLD);
		MPI_Send(&values[leaf_len / 2], leaf_len / 2, MPI_INT, child + 1, 0, MPI_COMM_WORLD);

		int *received[2];
		for(int i = 0; i < 2; i++){
			received[i] = malloc(leaf_len/2*sizeof(int));
			if(received[i] == NULL){
				printf("P%d: Not enough memory to allocate auxiliary array of length %d\n", rank, leaf_len/2);
				MPI_Finalize();
				exit(1);
			}
		}

		for(int i = 0; i < 2; i++){
			/* Receive from one of the childs */
			MPI_Recv(received[i], leaf_len / 2, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			#if DEBUG == 1
				printf("P%d: Received array from child %d\n", rank, status.MPI_SOURCE);
				print_array(received[i], leaf_len / 2);
			#endif
		}

		/* Place back the message in the array (interleaved) */
		interleave(received[0], received[1], values, leaf_len);

		for(int i = 0; i < 2; i++){
			free(received[i]);
			received[i] = NULL;
		}
	} else {
		/* Conquer! */
		#if DEBUG == 1
			printf("P%d: Sorting array\n", rank);
		#endif
		bubblesort(values, leaf_len);
	}

	if(rank == 0){
	#if DEBUG == 1
		/* Root node achieved. Show the array */
		print_array(values, leaf_len);
	# else
		double now = MPI_Wtime();
		printf("P%d: Array sorted in %f\n", rank, now - then);
	#endif
	} else {
		#if DEBUG == 1
			printf("P%d: Sending back to parent %d\n", rank, parent);
		#endif
		/* Send back array to parent */
		MPI_Send(values, leaf_len, MPI_INT, parent, 0, MPI_COMM_WORLD);
	}

	free(values);
	values = NULL;
	
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

void bubblesort(int *array, const int SIZE)
{
	/* Sort the array: bubblesort */
	bool swapped = true;
	for(int i = 0; swapped && i < SIZE; i++){
		swapped = false;
		for(int j = i + 1; j < SIZE; j++){
			if(array[j] < array[i]){
				int swap = array[i];
				array[i] = array[j];
				array[j] = swap;
				swapped = true;
			}
		}
	}
}

void interleave(int *src_a, int *src_b, int *dest, int length)
{
	int ia = 0;
	int ib = 0;
	for(int i = 0; i < length; i++){
		if(src_a[ia] <= src_b[ib] && ia < length / 2)
			dest[i] = src_a[ia++];
		else
			dest[i] = src_b[ib++];
	}
}

#if DEBUG == 1
void print_array(int *array, int len)
{
	printf("\n");
	printf("Array: ");
	for(int i = 0; i < len; i++)
		printf("%d ", array[i]);
	printf("\n");
}
#endif
