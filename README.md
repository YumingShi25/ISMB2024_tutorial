# Numba tutorial "Just-in-time compiled Python for bioinformatics research"

### Organizer:
Sven Rahmann

### Speakers:
Johanna Schmitz, Center for Bioinformatics Saar and Saarland University, Saarland Informatics Campus, Saarbrücken, Germany; Saarbrücken Graduate School of Computer Science

Jens Zentgraf, Center for Bioinformatics Saar and Saarland University, Saarland Informatics Campus, Saarbrücken, Germany; Saarbrücken Graduate School of Computer Science

Sven Rahmann, Center for Bioinformatics Saar and Saarland University, Saarland Informatics Campus, Saarbrücken, Germany

## Description
Python has a reputation for being a clean and easy-to-learn language, but slow when it comes to execution, and difficult concerning multi-threaded execution. Nonetheless, it is one of the most popular languages in science, including bioinformatics, because for many tasks, efficient libraries exist, and Python acts as a glue language. In this tutorial, we explore how to write efficient multi-threaded applications in Python using the numba just-in-time compiler. In this way, we can use Python’s flexibility and the existing packages to handle high-level functionality (e.g., design the user interface, run machine learning models), and then use compiled Python for additional custom compute-heavy tasks; these parts can even run in parallel.

In this tutorial, we introduce a small (but still interesting and relevant) problem as an example: efficient search for bipartite DNA motifs. We develop an efficient tool that outputs every match in a reference genome in a matter of seconds. Starting with an introduction to the problem and a (slow) pure Python implementation, we learn how to write more jit-compiler-friendly code, transition towards a compiled version and observe speed increases until we obtain C-like speed. We parallelize the tool to make it even faster, and add more options for more flexible searching. Finally, we add a simple but effective GUI, which can increase the potential user-base of such a tool by an order of magnitude.

## Learning Objectives

* Understand the difference between interpretation, lazy and eager/early compilation
* Understand the possibilities and limitations of the numba just-in-time compiler
* Explore several examples about when numba can accelerate your code (and when it cannot)
* Understand pre-requisites for compiling a function
* Learn the differences between compileable and non-compileable Python code
* Learn about parallelizing Python in spite of the Global Interpreter Lock (GIL) with compiled functions
* Learn how to scale up a prototype to handle large data
* Get an understanding of DNA motif search

## Getting started

### Installing the environment

To work through the tutorial, you need a recent Python version and several additional Python packages (they are listed in the environment.yml file in the main directory of the repository).

We recommend that you use the miniforge Python distribution which uses the conda and mamba package managers and allows you to set up separate environments for different projects. It also has the advantage that it works purely from user directories, i.e., you do not need admin access to the computer to install anything. Everything works from your home directory. 

We give some guidance for installing miniforge and using mamba here. If you use a different Python distribution, you have to install the required packages using your package manager. We recommend that you do not modify your system Python installation.

Miniforge can be found at [www.github.com/conda-forge/miniforge](www.github.com/conda-forge/miniforge). You will find downloads for each major operating system. 

After you successfully installed the Miniforge, you can create the mamba environment using the command

```
mamba env create
```

## Downloading T2T

We will later need the T2T genome for motif search. To download the fasta file run
``` 
./download_t2t.sh
```
To uncompress the file run
```
gzip -dk chm13v2.0.fa.gz
```