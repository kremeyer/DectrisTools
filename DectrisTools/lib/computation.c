#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

/* docstring definitions */
static char module_docstring[] =
    "fast and memory efficient array manipulation for diffraction data processing";
static char masked_sum_docstring[] =
    "compute the sums along axis 1 and 2 in a 3d array; a mask is applied before summation";

/* function declarations */
static PyObject *masked_sum (PyObject * self, PyObject * args);

/* module method definitions */
static PyMethodDef ComputationMethods[] = {
    {"masked_sum", masked_sum, METH_VARARGS, masked_sum_docstring},
    {NULL, NULL, 0, NULL},
};

/* module definition to be passed to the interpreter by the initialization function */
static struct PyModuleDef computation = {
    PyModuleDef_HEAD_INIT,
    "computation",
    module_docstring,
    -1,
    ComputationMethods
};

/* initialization function that passes the module definition to the interpreter */
PyMODINIT_FUNC
PyInit_computation (void)
{
    PyObject *m;
    m = PyModule_Create (&computation);
    import_array ();
    return m;
}


/**************************************/
/*  ACTUAL FUNCTIONALITY STARTS HERE  */
/**************************************/

static PyObject *
masked_sum (PyObject * self, PyObject * args)
{
    PyObject *sum_obj;
    PyArrayObject *sum_npyarray;
    PyObject *images_obj, *mask_obj;
    PyArrayObject *images_npyarray, *mask_npyarray;
    npy_intp *images_shape;
    npy_intp *mask_shape;
    npy_uint16 ***images, **mask;
    npy_uint64 *sum;

    /* parse input objects, check dimensions and data types */
    if (!PyArg_ParseTuple (args, "OO", &images_obj, &mask_obj))
	{
	    PyErr_SetString (PyExc_TypeError, "error parsing input");
	    return NULL;
	}

    images_npyarray = (PyArrayObject *) PyArray_FROM_O (images_obj);
    if (PyArray_NDIM (images_npyarray) != 3)
	{
	    PyErr_SetString (PyExc_IndexError,
			     "expected ndim=3 images array");
	    return NULL;
	}
    if (PyArray_TYPE (images_npyarray) != NPY_UINT16)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "expected uint16 images array");
	    return NULL;
	}

    mask_npyarray = (PyArrayObject *) PyArray_FROM_O (mask_obj);
    if (PyArray_NDIM (mask_npyarray) != 2)
	{
	    PyErr_SetString (PyExc_IndexError, "expected ndim=2 mask array");
	    return NULL;
	}
    if (PyArray_TYPE (mask_npyarray) != NPY_UINT16)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "expected uint16 mask array");
	    return NULL;
	}

    images_shape = PyArray_SHAPE (images_npyarray);
    mask_shape = PyArray_SHAPE (mask_npyarray);
    if (images_shape[1] != mask_shape[0] || images_shape[2] != mask_shape[1])
	{
	    PyErr_SetString (PyExc_IndexError,
			     "mask and image sizes do not match");
	    return NULL;
	}

    /* allocate memory for return array */
    sum_obj =
	PyArray_Zeros (1, &images_shape[0],
		       PyArray_DescrFromType (NPY_UINT64), 0);
    sum_npyarray = (PyArrayObject *) sum_obj;

    /* get pointers that simulate C-style array for easier iteration */
    if (PyArray_AsCArray
	((PyObject **) & images_npyarray, (void *) &images, images_shape, 3,
	 PyArray_DescrFromType (NPY_UINT16)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "conversion of images to c array failes");
	    return NULL;
	}
    if (PyArray_AsCArray
	((PyObject **) & mask_npyarray, (void *) &mask, mask_shape, 2,
	 PyArray_DescrFromType (NPY_UINT16)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "conversion of mask to c array failes");
	    return NULL;
	}

    npy_intp *n_imgs[1];
    n_imgs[0] = &images_shape[0];
    if (PyArray_AsCArray
	((PyObject **) & sum_npyarray, (void *) &sum, *n_imgs, 1,
	 PyArray_DescrFromType (NPY_UINT64)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "conversion of sums to c array failes");
	    return NULL;
	}

//    Py_BEGIN_ALLOW_THREADS;

    /* actual computation */
    for (int i = 0; i < images_shape[0]; i++)
	{
	    for (int j = 0; j < images_shape[1]; j++)
		{
		    for (int k = 0; k < images_shape[2]; k++)
			{
			    sum[i] += images[i][j][k]; // * mask[j][k];
			}
		}
	}

//    Py_END_ALLOW_THREADS;

    /* keep track of the reference counting to make sure the python garbage collection can do it's thing */
//    Py_DECREF (images_obj);
//    Py_DECREF (mask_obj);
    PyArray_Free(images_obj, (void*)images);
    PyArray_Free(mask_obj, (void*)mask);
    PyArray_Free(sum_obj, sum);
//    Py_DECREF (images_obj);
//    Py_DECREF (mask_obj);
//    free(images);
//    free(mask);
//    printf("%li %li\n", mask_shape[0], mask_shape[1]);
    return sum_obj;
}
