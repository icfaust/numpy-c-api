#include <Python.h>
#include <stdint.h>

#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>


void _fib(int64_t n, int64_t* ar_ptr)
{
    for(int64_t j = 0; j<n; j++)
    {
        if(j<2)
        {
            ar_ptr[j] = j;
        }
        else
        {
            ar_ptr[j] = ar_ptr[j-1] + ar_ptr[j-2];
        }
    }
}

static PyObject* fib_new_array(PyObject* self, PyObject* args)
{
    int64_t n, *ar_ptr;
    PyObject* output;

    /* Parse the input, from Python integer to C int64_t */
    if (!PyArg_ParseTuple(args, "l", &n))
        return NULL;

    npy_intp size = n;

    /* Create a 1-D C contiguous NumPy Int64 array of size n */
    output = PyArray_NewFromDescr(&PyArray_Type, 
                                  PyArray_DescrFromType(NPY_INT64),
                                  1,
                                  &size,
                                  NULL,
                                  NULL,
                                  NPY_ARRAY_OWNDATA,
                                  NULL);

    ar_ptr = (int64_t *) PyArray_DATA(output);

    _fib(n, ar_ptr);

    return output;
}

static PyObject* fib_in_place_array(PyObject* self, PyObject* args)
{
    int64_t n, *ar_ptr;
    PyObject* output = NULL;

    /* Parse the input, from Numpy array */
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &output))
        return NULL;
    
    /* Assume data is contiguous and run through as if it is a 1-D array */
    n = PyArray_SIZE(output);
    ar_ptr = (int64_t *) PyArray_DATA(output);

    _fib(n, ar_ptr);

    Py_RETURN_NONE;
}

/* Wrapped functions */
static PyObject* fib(PyObject* self, PyObject* args)
{
    int64_t n;
    PyObject* output = NULL;

    if (PyArg_ParseTuple(args, "l", &n))
    {
        return fib_new_array(self, args);
    }
    
    PyErr_Clear();
    
    if (PyArg_ParseTuple(args, "O!", &PyArray_Type, &output))
    {
        return fib_in_place_array(self, args);
    }

    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */
    return NULL;
}

/* Define functions in module */
static PyMethodDef FibMethods[] = {
    {"fib", fib, METH_VARARGS, "Calculate the Fibonacci numbers (in C)."},
    {"fibNewArray", fib_new_array, METH_VARARGS, "Generate a Fibonacci number array (in C returning a numpy array)."},
    {"fibInPlace", fib_in_place_array, METH_VARARGS, "Generate a Fibonacci number array in place (in C returning a numpy array)."},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Create PyModuleDef structure */
static struct PyModuleDef fibStruct = {
    PyModuleDef_HEAD_INIT,
    "fib",
    "",
    -1,
    FibMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

/* Module initialization */
PyObject *PyInit_fib(void)
{
    import_array();
    return PyModule_Create(&fibStruct);
}