#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

/* docstring definitions */
static char module_docstring[] =
    "fast and memory efficient array manipulation for diffraction data processing";
static char masked_sum_docstring[] =
    "compute the sums along axis 1 and 2 in a 3d array; a mask is applied before summation";
static char normed_stack_docstring[]=
    "normalize stack of images to an 1d array";

/* function declarations */
static PyObject *masked_sum (PyObject * self, PyObject * args);
static PyObject *normed_stack (PyObject * self, PyObject * args);

/* module method definitions */
static PyMethodDef ComputationMethods[] = {
    {"masked_sum", masked_sum, METH_VARARGS, masked_sum_docstring},
    {"normed_stack", normed_stack, METH_VARARGS, normed_stack_docstring},
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
	    Py_XDECREF(images_npyarray);
	    return NULL;
	}
    if (PyArray_TYPE (images_npyarray) != NPY_UINT16)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "expected uint16 images array");
	    Py_XDECREF(images_npyarray);
	    return NULL;
	}

    mask_npyarray = (PyArrayObject *) PyArray_FROM_O (mask_obj);
    if (PyArray_NDIM (mask_npyarray) != 2)
	{
	    PyErr_SetString (PyExc_IndexError, "expected ndim=1 mask array");
	    Py_XDECREF(mask_npyarray);
	    Py_XDECREF(images_npyarray);
	    return NULL;
	}
    if (PyArray_TYPE (mask_npyarray) != NPY_UINT16)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "expected uint16 mask array");
	    Py_XDECREF(mask_npyarray);
	    Py_XDECREF(images_npyarray);
	    return NULL;
	}

    images_shape = PyArray_SHAPE (images_npyarray);
    mask_shape = PyArray_SHAPE (mask_npyarray);
    if (images_shape[1] != mask_shape[0] || images_shape[2] != mask_shape[1])
	{
	    PyErr_SetString (PyExc_IndexError,
			     "mask and image sizes do not match");
	    Py_XDECREF(mask_npyarray);
	    Py_XDECREF(images_npyarray);
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
			     "conversion of images to c array failed");
	    Py_XDECREF(mask_npyarray);
	    Py_XDECREF(images_npyarray);
	    return NULL;
	}
    if (PyArray_AsCArray
	((PyObject **) & mask_npyarray, (void *) &mask, mask_shape, 2,
	 PyArray_DescrFromType (NPY_UINT16)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "conversion of mask to c array failed");
        PyArray_Free(images_obj, images);
	    Py_XDECREF(mask_npyarray);
	    Py_XDECREF(images_npyarray);
	    return NULL;
	}

    npy_intp *n_imgs[1];
    n_imgs[0] = &images_shape[0];
    if (PyArray_AsCArray
	((PyObject **) & sum_npyarray, (void *) &sum, *n_imgs, 1,
	 PyArray_DescrFromType (NPY_UINT64)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "conversion of sums to c array failed");
        PyArray_Free(mask_obj, mask);
        PyArray_Free(images_obj, images);
	    Py_XDECREF(mask_npyarray);
	    Py_XDECREF(images_npyarray);
	    return NULL;
	}

    Py_BEGIN_ALLOW_THREADS;

    /* actual computation */
    for (int i = 0; i < images_shape[0]; i++)
	{
	    for (int j = 0; j < images_shape[1]; j++)
		{
		    for (int k = 0; k < images_shape[2]; k++)
			{
			    sum[i] += images[i][j][k] * mask[j][k];
			}
		}
	}

    Py_END_ALLOW_THREADS;

    /* keep track of the reference counting to make sure the python garbage collection can do it's thing */
    PyArray_Free(images_obj, images);
    PyArray_Free(mask_obj, mask);
    PyArray_Free(sum_obj, sum);
    Py_DECREF (images_npyarray);
    Py_DECREF (mask_npyarray);

    return sum_obj;
}


static PyObject *
normed_stack (PyObject * self, PyObject * args)
{
    PyObject *normed_images_obj;
    PyArrayObject *normed_images_npyarray;
    PyObject *images_obj, *norm_values_obj;
    PyArrayObject *images_npyarray, *norm_values_npyarray;
    npy_intp *images_shape;
    npy_intp *norm_values_shape;
    npy_uint16 ***images;
    npy_float32 ***normed_images, *norm_values;

    /* parse input objects, check dimensions and data types */
    if (!PyArg_ParseTuple (args, "OO", &images_obj, &norm_values_obj))
	{
	    PyErr_SetString (PyExc_TypeError, "error parsing input");
	    return NULL;
	}

    images_npyarray = (PyArrayObject *) PyArray_FROM_O (images_obj);
    if (PyArray_NDIM (images_npyarray) != 3)
	{
	    PyErr_SetString (PyExc_IndexError,
			     "expected ndim=3 images array");
	    Py_XDECREF(images_npyarray);
	    return NULL;
	}
    if (PyArray_TYPE (images_npyarray) != NPY_UINT16)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "expected uint16 images array");
	    Py_XDECREF(images_npyarray);
	    return NULL;
	}

    norm_values_npyarray = (PyArrayObject *) PyArray_FROM_O (norm_values_obj);
    if (PyArray_NDIM (norm_values_npyarray) != 1)
	{
	    PyErr_SetString (PyExc_IndexError, "expected ndim=1 norm_values array");
        Py_XDECREF(images_npyarray);
	    Py_XDECREF(norm_values_npyarray);
	    return NULL;
	}
    if (PyArray_TYPE (norm_values_npyarray) != NPY_FLOAT32)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "expected float32 norm_values array");
        Py_XDECREF(images_npyarray);
	    Py_XDECREF(norm_values_npyarray);
	    return NULL;
	}

    images_shape = PyArray_SHAPE (images_npyarray);
    norm_values_shape = PyArray_SHAPE (norm_values_npyarray);
    if (images_shape[0] != norm_values_shape[0])
	{
	    PyErr_SetString (PyExc_IndexError,
			     "normed_values and image sizes do not match");
        Py_XDECREF(images_npyarray);
	    Py_XDECREF(norm_values_npyarray);
	    return NULL;
	}

    /* allocate memory for return array */
    normed_images_obj =
	PyArray_Empty (3, images_shape,
		       PyArray_DescrFromType (NPY_FLOAT32), 0);
    normed_images_npyarray = (PyArrayObject *) normed_images_obj;

    /* get pointers that simulate C-style array for easier iteration */
    if (PyArray_AsCArray
	((PyObject **) & images_npyarray, (void *) &images, images_shape, 3,
	 PyArray_DescrFromType (NPY_UINT16)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "conversion of images to c array failed");
        Py_XDECREF(images_npyarray);
	    Py_XDECREF(norm_values_npyarray);
	    return NULL;
	}
    if (PyArray_AsCArray
	((PyObject **) & norm_values_npyarray, (void *) &norm_values, norm_values_shape, 1,
	 PyArray_DescrFromType (NPY_FLOAT32)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "conversion of norm_values to c array failed");
        PyArray_Free(images_obj, (void*)images);
        Py_XDECREF(images_npyarray);
	    Py_XDECREF(norm_values_npyarray);
	    return NULL;
	}

    if (PyArray_AsCArray
	((PyObject **) & normed_images_npyarray, (void *) &normed_images, images_shape, 3,
	 PyArray_DescrFromType (NPY_FLOAT32)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "conversion of normed_images to c array failed");
        PyArray_Free(images_obj, (void*)images);
        PyArray_Free(norm_values_obj, (void*)norm_values);
        Py_XDECREF(images_npyarray);
	    Py_XDECREF(norm_values_npyarray);
	    return NULL;
	}

	Py_BEGIN_ALLOW_THREADS;

    /* actual computation */
    for (int i = 0; i < images_shape[0]; i++)
	{
	    for (int j = 0; j < images_shape[1]; j++)
		{
		    for (int k = 0; k < images_shape[2]; k++)
			{
			    normed_images[i][j][k] += images[i][j][k] / norm_values[i];
			}
		}
	}

	Py_END_ALLOW_THREADS;

    /* keep track of the reference counting to make sure the python garbage collection can do it's thing */
    PyArray_Free(images_obj, images);
    PyArray_Free(norm_values_obj, norm_values);
    PyArray_Free(normed_images_obj, normed_images);
    Py_DECREF (images_npyarray);
    Py_DECREF (norm_values_npyarray);
    return normed_images_obj;
}