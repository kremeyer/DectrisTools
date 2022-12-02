#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>


static PyObject *
masked_sum (PyObject * self, PyObject * args)
{
    PyObject *sum_obj;
    PyArrayObject *sum_array;
    PyObject *images_obj, *mask_obj;
    PyArrayObject *images_array, *mask_array;

    if (!PyArg_ParseTuple (args, "OO", &images_obj, &mask_obj))
	{
	    PyErr_SetString (PyExc_TypeError, "error parsing input");
	    return NULL;
	}

    images_array = (PyArrayObject *) PyArray_FROM_O (images_obj);
    if (PyArray_NDIM (images_array) != 3)
	{
	    PyErr_SetString (PyExc_IndexError,
			     "expected ndim=3 images array");
	    return NULL;
	}
    if (PyArray_TYPE (images_array) != NPY_UINT16)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "expected uint16 images array");
	    return NULL;
	}
    npy_intp *images_shape = PyArray_SHAPE (images_array);

    mask_array = (PyArrayObject *) PyArray_FROM_O (mask_obj);
    if (PyArray_NDIM (mask_array) != 2)
	{
	    PyErr_SetString (PyExc_IndexError, "expected ndim=2 mask array");
	    return NULL;
	}
    if (PyArray_TYPE (mask_array) != NPY_UINT16)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "expected uint16 mask array");
	    return NULL;
	}
    npy_intp *mask_shape = PyArray_SHAPE (mask_array);

    if (images_shape[1] != mask_shape[0] || images_shape[2] != mask_shape[1])
	{
	    PyErr_SetString (PyExc_IndexError,
			     "mask and image sizes do not match");
	    return NULL;
	}

    sum_obj =
	PyArray_Zeros (1, &images_shape[0],
		       PyArray_DescrFromType (NPY_UINT64), 0);
    sum_array = (PyArrayObject *) sum_obj;

    uint16_t ***images, **mask;
    if (PyArray_AsCArray
	((PyObject **) & images_array, (void *) &images, images_shape, 3,
	 PyArray_DescrFromType (NPY_UINT16)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "conversion of images to c array failes");
	    return NULL;
	}
    if (PyArray_AsCArray
	((PyObject **) & mask_array, (void *) &mask, mask_shape, 2,
	 PyArray_DescrFromType (NPY_UINT16)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "conversion of mask to c array failes");
	    return NULL;
	}

    uint64_t *sum;
    npy_intp *n_imgs[1];
    n_imgs[0] = &images_shape[0];
    if (PyArray_AsCArray
	((PyObject **) & sum_array, (void *) &sum, *n_imgs, 1,
	 PyArray_DescrFromType (NPY_UINT64)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError,
			     "conversion of sums to c array failes");
	    return NULL;
	}

    Py_BEGIN_ALLOW_THREADS

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

    Py_END_ALLOW_THREADS

    Py_DECREF(images_obj);
    Py_DECREF(images_obj);
    Py_DECREF(sum_obj);
    Py_DECREF(mask_obj);
    Py_DECREF(mask_obj);

    return sum_obj;
}

static PyMethodDef ComputationMethods[] = {
    {"masked_sum", masked_sum, METH_VARARGS,
     "Get sum of images along 2nd and 3rd axis with a mask applied to each image"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef computation = {
    PyModuleDef_HEAD_INIT,
    "computation",
    NULL,			// docstring
    -1,
    ComputationMethods
};

PyMODINIT_FUNC
PyInit_computation (void)
{
    PyObject *m;
    m = PyModule_Create (&computation);
    import_array ();
    return m;
}
