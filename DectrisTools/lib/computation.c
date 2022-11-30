#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *
masked_sum(PyObject *self, PyObject *args)
{
    PyObject *images_obj, mask_obj;

    if (!PyArg_ParseTuple(args, "OO", &images_obj, &mask_obj)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing input");
    return NULL;
    }
}

static PyMethodDef ComputationMethods[] =
{
    {"masked_sum", masked_sum, METH_VARARGS, "Get sum of images along 2nd and 3rd axis with a mask applied to each image"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef computation =
{
    PyModuleDef_HEAD_INIT,
    "computation",
    NULL, // docstring
    -1,
    ComputationMethods
};

PyMODINIT_FUNC
PyInit_computation(void)
{
    return PyModule_Create(&computation);
}