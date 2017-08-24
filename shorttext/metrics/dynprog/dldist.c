
#include "dldist.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#include <string.h>

int damerau_levenshtein(char *word1, char *word2)
{
  int len1 = strlen(word1);
  int len2 = strlen(word2);
  int i, j;

  int matrix[len1+1][len2+1];
  for (i=0; i<=len1; i++) matrix[i][0] = i;
  for (j=0; j<=len2; j++) matrix[0][j] = j;

  for (i=1; i<=len1; i++) {
    for (j=1; j<=len2; j++) {
      int cost = 0;
      if (word1[i]!=word2[j]) cost = 1;
      int delcost = matrix[i-1][j] + 1;
      int inscost = matrix[i][j-1] + 1;
      int subcost = matrix[i-1][j-1] + cost;
      int score = MIN(MIN(delcost, inscost), subcost);
      if ((i>1) & (j>1) & (word1[i]==word2[j-1]) & (word1[i-1]==word2[j])) {
	    score = MIN(score, matrix[i-2][j-2]+cost);
      }
      matrix[i][j] = score;
    }
  }

  return(matrix[len1][len2]);
}

int longest_common_prefix(char *word1, char *word2) {
  int len1 = strlen(word1);
  int len2 = strlen(word2);

  int lcp = 0;
  int i;

  for (i=0; i<MIN(len1, len2); i++) {
    if (word1[i]==word2[i]) lcp++; else break;
  }

  return lcp;
}
