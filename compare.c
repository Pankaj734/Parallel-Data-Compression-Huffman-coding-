#include<stdio.h>
#include<stdlib.h>

int main(int argc, char ** argv)
{
    if(argc != 3){
        printf("Enter original file and decompressed file.");
        return -1;
    }
    FILE * OG = fopen(argv[1], "rb");
    FILE * Decomp = fopen(argv[2], "rb");

    char c1 , c2;
    while(c1 != EOF && c2 != EOF){
        c1 = fgetc(OG);
        c2 = fgetc(Decomp);

        if(c1 != c2){
            printf("Files don't Match.");
            return -1;
        }
    }
    
    printf("Files Matched. Bravo.");
    return 0;
}