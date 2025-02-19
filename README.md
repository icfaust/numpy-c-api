NumPy C-API quick start
=========================

A quick example of Python 3.10 modules implemented in C using the Python/C and NumPy APIs.

```
make all
python -c "import hello, numpy; hello.greet(numpy.array('World'))"
python -c "import fib, numpy; print(fib.fib(5))"
make clean
```

Note that the best way to distribute a Python module is to use a `setup.py`
file. The Makefile is here only to debunk Python's magic.


## Steps

1. Write a `foobarmodule.c` C file.
   Note the `#include <Python.h>`, `#include <numpy/numpyconfig.h>`, and `#include <numpy/arrayobject.h>` at the beginning of example files.
2. Compile `foobarmodule.c` and produce a `foobar.so` file.
   See the Makefile for details.
3. Put `foobar.so` in a folder listed in the Python path.
4. Load the module from Python with `import foobar`.
