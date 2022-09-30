
#include <stdio.h>

// find all possible combinations of a set of numbers
// and print them out
// the numbers are 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
// the number of combinations is 2^10 = 1024
void printCombinations(int n, int *a, int cur)
{
    int i;
    if (cur == n) {
        // we have a complete combination
        for (i = 0; i < n; i++)
            printf("%d", a[i]);
        printf("\n");
    } else {
        // try both possibilities at this step
        a[cur] = 0;
        printCombinations(n, a, cur+1);
        a[cur] = 1;
        printCombinations(n, a, cur+1);
    }
}

int main()
{
    int a[10];
    printCombinations(10, a, 0);
    return 0;
}