# GPU Programming course

GPU programming course solution created by Anton Backman in CUDA that calculated angles between galaxies

***

Program is compiled on CUDA supported computers with the command:

```
nvcc -o <compiled_program_name> GPU_Programming.cu
```

***

In VSCode you can also run the build included Build Task with Ctrl+Shift+B
(In case of error, make sure path to nvcc is correct in tasks.json)

***

To run the compiled program you need two input files containing galaxies, example run command:

```
.\<compiled_program_name> .\<path_to_real_galaxy> .\<path_to_simulated_galaxy>
```

***

The program will output 4 .txt files with the results.
