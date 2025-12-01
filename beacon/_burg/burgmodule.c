#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

void burg(int n, const double* x, int pmax, double* coefs, double* var1, double* var2);

// Python wrapper
static PyObject* py_burg(PyObject* self, PyObject* args) {
    PyArrayObject *x_in = NULL;
    int pmax;
    
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &x_in, &pmax)) {
        return NULL;
    }
    
    if (PyArray_TYPE(x_in) != NPY_DOUBLE || PyArray_NDIM(x_in) != 1) {
        PyErr_SetString(PyExc_TypeError, "Input array must be 1D double.");
        return NULL;
    }
    
    int n = (int) PyArray_SIZE(x_in);
    double* x_data = (double*) PyArray_DATA(x_in);
    
    // Output arrays
    npy_intp coefs_dims[2] = {pmax, pmax};
    npy_intp var_dims[1] = {pmax + 1};
    
    PyObject* coefs_out = PyArray_ZEROS(2, coefs_dims, NPY_DOUBLE, 0);
    PyObject* var1_out  = PyArray_ZEROS(1, var_dims, NPY_DOUBLE, 0);
    PyObject* var2_out  = PyArray_ZEROS(1, var_dims, NPY_DOUBLE, 0);
    
    burg(
        n,
        x_data,
        pmax,
        (double*) PyArray_DATA((PyArrayObject*) coefs_out),
        (double*) PyArray_DATA((PyArrayObject*) var1_out),
        (double*) PyArray_DATA((PyArrayObject*) var2_out)
    );
    
    return Py_BuildValue("NNN", coefs_out, var1_out, var2_out);
}

// Method table
static PyMethodDef BurgMethods[] = {
    {"burg", py_burg, METH_VARARGS, "R-style Burg AR estimation."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef burgmodule = {
    PyModuleDef_HEAD_INIT,
    "burg",
    NULL,
    -1,
    BurgMethods
};

// Init
PyMODINIT_FUNC PyInit_burg(void) {
    import_array();
    return PyModule_Create(&burgmodule);
}
