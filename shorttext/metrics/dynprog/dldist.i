
%module dldist

%{
#define SWIG_FILE_WITH_INIT
#include "dldist.h"
%}

int damerau_levenshtein(char *word1, char *word2);
