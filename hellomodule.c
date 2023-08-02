#include <Python.h>

#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>


static PyObject* greet(PyObject* self, PyObject* args)
{
    const wchar_t* name;
    int length;
    PyObject* output = NULL;

    /* Parse the input, from Numpy array string to C string */
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &output))
        return NULL;
    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */

    /* Unicode NumPy arrays are not \0 terminated */
    name = (wchar_t *) PyArray_DATA(output);
    length = PyArray_ITEMSIZE(output)/sizeof(wchar_t);

    printf("Hello %*.*ls\n", length, length, name);

    /* Returns a None Python object */
    Py_RETURN_NONE;
}

/* Define functions in module */
static PyMethodDef HelloMethods[] = {
    {"greet", greet, METH_VARARGS, "Greet somebody (in C)."},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Create PyModuleDef stucture */
static struct PyModuleDef helloStruct = {
    PyModuleDef_HEAD_INIT,
    "hello",
    "",
    -1,
    HelloMethods,
    NULL,
    NULL,
    NULL,
    NULL
};


/* Module initialization */
PyObject *PyInit_hello(void)
{
    import_array();
    return PyModule_Create(&helloStruct);
}