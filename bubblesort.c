#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/* Define DEBUG by compiler's command line (-DDEBUG=1) */
#ifndef DEBUG
	#define DEBUG 0
#endif

/* If should use qsort instead of bubblesort (-DQSORT=1) */
#ifndef QSORT
	#define QSORT 0
#endif

void bubblesort(int *array, const int SIZE);
void combine(int *src_a, int len_a, int *src_b, int len_b, int *dst, int length);
#if DEBUG == 1
void print_array(int *array, int len);
#endif
#if QSORT == 1
int cmpfunc(const void *a, const void *b);
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


	/* Default with 1.000.000 entries */
	/* Or configure with compiler's command line (-DN=N) */
#ifndef N
	#if DEBUG == 1
		/* DEBUG array with 40 entries */
		const unsigned ROOT_LEN = 40;
	#else
		const unsigned ROOT_LEN = 1000000;
	#endif
#else
	const unsigned ROOT_LEN = N;
#endif

	MPI_Status status;
	int *values = NULL;
	int node_len = ROOT_LEN;
	int parent = -1;

#if DEBUG == 0
	double then = 0;
#endif
	if(rank == 0){
		/* Root: initial producer */

		/* Allocate memory for the whole array */
		values = malloc(node_len*sizeof(int));
		if(values == NULL){
			printf("P%d: Not enough memory to allocate main array of length %d\n", rank, node_len);
			MPI_Finalize();
			exit(1);
		}

		/* Populate array in decreasing order to execute worst case */
	#if DEBUG == 1
		printf("P%d: Populating main array\n", rank);
	#endif
		for(int i = 0; i < node_len; i++)
			values[i] = node_len - i;

	printf("P%d: Sorting array of %d elements in %d process\n", rank, node_len, size);
	#if DEBUG == 1
		print_array(values, node_len);
	#else
		/* Measure time when not in DEBUG mode */
		then = MPI_Wtime();
	#endif

	} else {
		/* Check the size of the incoming array */
		MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_INT, &node_len);
		parent = status.MPI_SOURCE;
	#if DEBUG == 1
		printf("P%d: Incoming array of size %d. Allocating memory\n", rank, node_len);
	#endif

		/* Allocate only the necessary memory */
		values = malloc(node_len*sizeof(int));
		if(values == NULL){
			printf("P%d: Not enough memory to allocate main array of length %d\n", rank, node_len);
			MPI_Finalize();
			exit(1);
		}

	#if DEBUG == 1
		printf("P%d: Receiving array from parent %d\n", rank, parent);
	#endif
		/* Finally receive from parent */ 
		MPI_Recv(values, node_len, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
	}

	/* Check the last child existance */
	int first_child = rank*2 + 1;

	if(first_child + 1 < size){
		/* Divide to childs */
	#if DEBUG == 1
		printf("P%d: Sending array to childs %d and %d\n", rank, first_child, first_child + 1);
	#endif
		int half_len = node_len / 2;

		/* No need to block the send. It will only receive the answer once sucessfuly sent */
		MPI_Request request[2];
		MPI_Isend(values, half_len, MPI_INT, first_child, 0, MPI_COMM_WORLD, &request[0]);
		/* Send the remaining values (useful if node_len is odd) */
		MPI_Isend(&values[half_len], node_len - half_len, MPI_INT, first_child + 1, 0, MPI_COMM_WORLD, &request[1]);

		int *received[2];
		received[0] = malloc(half_len*sizeof(int));
		if(received[0] == NULL){
			printf("P%d: Not enough memory to allocate auxiliary array of length %d\n", rank, half_len);
			MPI_Finalize();
			exit(1);
		}

		received[1] = malloc((node_len - half_len)*sizeof(int));
		if(received[1] == NULL){
			printf("P%d: Not enough memory to allocate auxiliary array of length %d\n", rank, node_len - half_len);
			MPI_Finalize();
			exit(1);
		}

		for(int i = 0; i < 2; i++){
			/* Receive from one of the childs */
			MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			if(status.MPI_SOURCE == first_child){
				MPI_Recv(received[0], half_len, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
			#if DEBUG == 1
				printf("P%d: Received array from child %d\n", rank, status.MPI_SOURCE);		
				print_array(received[0], half_len);
			#endif
			} else {
				MPI_Recv(received[1], node_len - half_len, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
			#if DEBUG == 1
				printf("P%d: Received array from child %d\n", rank, status.MPI_SOURCE);		
				print_array(received[1], node_len - half_len);
			#endif
			}
		}

		/* Place back the message in the array (combined) */
		combine(received[0], half_len, received[1], node_len - half_len, values, node_len);

		for(int i = 0; i < 2; i++){
			free(received[i]);
			received[i] = NULL;
		}
	} else {
		/* Conquer! */
		#if DEBUG == 1
			printf("P%d: Sorting array\n", rank);
		#endif
		#if QSORT == 1
			qsort(values, node_len, sizeof(int), cmpfunc);
		#else
			bubblesort(values, node_len);
		#endif
	}

	if(rank == 0){
	#if DEBUG == 1
		/* Root node achieved. Show the array */
		print_array(values, node_len);
	# else
		double now = MPI_Wtime();
		printf("P%d: Array sorted in %f\n", rank, now - then);
	#endif
	} else {
		#if DEBUG == 1
			printf("P%d: Sending back to parent %d\n", rank, parent);
		#endif
		/* Send back array to parent */
		/* This send can be blocking because there is no more computation after this */
		MPI_Send(values, node_len, MPI_INT, parent, 0, MPI_COMM_WORLD);
	}

	free(values);
	values = NULL;
	
	MPI_Finalize();
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

void combine(int *src_a, int len_a, int *src_b, int len_b, int *dst, int length)
{
	int ia = 0;
	int ib = 0;
	for(int i = 0; i < length; i++){
		if((ia < len_a && src_a[ia] <= src_b[ib]) || ib == len_b)
			dst[i] = src_a[ia++];
		else
			dst[i] = src_b[ib++];
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

#if QSORT == 1
int cmpfunc(const void *a, const void *b)
{
	return *(int*)a - *(int*)b;
}
#endif
