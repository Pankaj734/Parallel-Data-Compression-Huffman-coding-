## Parallel Data Compression using Huffman Coding

### Files :
* **compress.cu** - Perform huffman compression on input file.
* **decompress.cu** - Perform huffman decompression on compressed file.
* **GenerateInput.cu** - Automatically generates required input file of specified length.
* **makefile** - Makefile
* **testcases** - Folder with multiple testcases.
* **README.md** - Brief description about project

### Quick Run :
* make - build and maintain groups of programs and files from the source code
* make clean - get rid of your object and executable files
* ./compress \<INPUT-FILENAME\> - compress given \<INPUT-FILENAME\>
* ./decompress \<COMPRESSED-FILENAME\> -decompress given \<COMPRESSED-FILENAME\>

### Steps for Execution -
* Build Histogram by calculating count of characters in given Input file.
* Recursively find 2 minimum frequencies to evaluate new frequency node.
* Build Huffman Tree and Huffman Dictionary using calculated frequency count.
* Compress given Input file using Huffman Dictionary.
* Store compressed data to output file.

### Unique stratergies -
* Build Histogram using parallel approach.
* Find 2 minimum frequencies using methodology similar to parallel reduction.
* Parallelly compress Input file along multiple threads.
* Moreover also build Huffman dictionary to avoid multiple traversal through Huffman tree.

### Future scope -
* To further optimize runtime and enhance effectiveness we can use following approaches -
  * Parallelized Burrows-Wheeler transform(BWT)
  * Parallelized Move-to-front transform (MTF)
 
### References -
* https://ieeexplore.ieee.org/document/6339599