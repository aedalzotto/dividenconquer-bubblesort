#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/* Define DEBUG by compiler's command line (-DDEBUG=1) */
#ifndef DEBUG
	#define DEBUG 0
#endif

unsigned last_pow_2(unsigned x);
void bubblesort(int *array, const int SIZE);
void interleave(int *src, int *dst, int length);
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

	/* Make the rank even */
	unsigned evenrank = rank % 2 ? rank + 1 : rank;

	/* Number of leaves in the tree depth */
	unsigned depth_len = rank ? last_pow_2(evenrank) : 1;

	/* Find the size of the array that will be computed in this node */
	unsigned leaf_len = ROOT_LEN / depth_len;

	/* Check if the node is the rightmost on its depth */
	bool is_right_child = !(rank % 2) && rank != 0;

	/* Find the parent of the leaf. This will be invalidated for root (node 0) */
	int parent = evenrank / 2 - 1;

	/* Find first child of the node. Will be invalidated if child >= size */
	int child = (rank * 2) + 1;

	/* Childs to the right can receive an extra value due to odd arrays */ 
	int *values = malloc((leaf_len + is_right_child)*sizeof(int));
	if(values == NULL){
		printf("P%d: Not enough memory to allocate main array of length %d\n", rank, leaf_len + is_right_child);
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
		for(int i = 0; i < ROOT_LEN; i++)
			values[i] = ROOT_LEN - i;

	printf("P%d: Sorting array of %d elements in %d process\n", rank, ROOT_LEN, size);
	#if DEBUG == 1
		print_array(values, ROOT_LEN);
	#else
		then = MPI_Wtime();
	#endif
	} else {
	#if DEBUG == 1
		printf("P%d: Receiving array from parent %d\n", rank, parent);
	#endif
		/* Childs to the right can receive an extra value due to odd arrays */ 
		MPI_Recv(values, leaf_len + is_right_child, MPI_INT, parent, 0, MPI_COMM_WORLD, &status);

		/* Update this node number of elements */
		MPI_Get_count(&status, MPI_INT, &leaf_len);
	}

	/* Check if number of elements in this node is odd */
	bool is_odd = leaf_len % 2;

	if(child + 1 < size){
		/* Divide to childs */
	#if DEBUG == 1
		printf("P%d: Sending array to childs %d and %d\n", rank, child, child + 1);
	#endif
		MPI_Send(values, leaf_len / 2, MPI_INT, child, 0, MPI_COMM_WORLD);
		MPI_Send(&values[leaf_len / 2], leaf_len / 2 + is_odd, MPI_INT, child + 1, 0, MPI_COMM_WORLD);

		int *received = malloc(leaf_len*sizeof(int));
		if(received == NULL){
			printf("P%d: Not enough memory to allocate auxiliary array of length %d\n", rank, leaf_len);
			MPI_Finalize();
			exit(1);
		}

		/* Receive from one of the childs */
		MPI_Recv(received, leaf_len / 2, MPI_INT, child, 0, MPI_COMM_WORLD, &status);
		#if DEBUG == 1
			printf("P%d: Received array from child %d\n", rank, status.MPI_SOURCE);
			print_array(received, leaf_len / 2);
		#endif

		MPI_Recv(&received[leaf_len / 2], leaf_len / 2 + is_odd, MPI_INT, child + 1, 0, MPI_COMM_WORLD, &status);
		#if DEBUG == 1
			printf("P%d: Received array from child %d\n", rank, status.MPI_SOURCE);
			print_array(&received[leaf_len / 2], leaf_len / 2 + is_odd);
		#endif

		/* Place back the message in the array (interleaved) */
		interleave(received, values, leaf_len);

		free(received);
		received = NULL;
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

void interleave(int *src, int *dst, int length)
{
	int ia = 0;
	int ib = length / 2;
	for(int i = 0; i < length; i++){
		if(ia < length / 2 && src[ia] <= src[ib] || ib == length)
			dst[i] = src[ia++];
		else
			dst[i] = src[ib++];
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
