#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

/* docstring definitions */
static char module_docstring[] = "fast and memory efficient array manipulation for diffraction data processing";
static char masked_histogram_docstring[] =
    "histogram for stack of uint16 images; a mask is applied before adding pixel to histogram";
static char masked_sum_docstring[] =
    "compute the sums along axis 1 and 2 in a 3d array; a mask is applied before summation";
static char normed_sum_docstring[] =
    "normalize stack of images to an 1d array and sum along the first axis; will return an image";

/* function declarations */
static PyObject *masked_histogram (PyObject * self, PyObject * args);
static PyObject *masked_sum (PyObject * self, PyObject * args);
static PyObject *normed_sum (PyObject * self, PyObject * args);

/* module method definitions */
static PyMethodDef ComputationMethods[] = {
    {"masked_histogram", masked_histogram, METH_VARARGS,
     masked_histogram_docstring},
    {"masked_sum", masked_sum, METH_VARARGS, masked_sum_docstring},
    {"normed_sum", normed_sum, METH_VARARGS, normed_sum_docstring},
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
masked_histogram (PyObject * self, PyObject * args)
{
    PyObject *histogram_obj;
    PyArrayObject *histogram_npyarray;
    PyObject *images_obj, *mask_obj;
    PyArrayObject *images_npyarray, *mask_npyarray;
    npy_intp *images_shape;
    npy_intp *mask_shape;
    npy_uint16 ***images, **mask;
    npy_uint64 *histogram;

    /* parse input objects, check dimensions and data types */
    if (!PyArg_ParseTuple (args, "OO", &images_obj, &mask_obj))
	{
	    PyErr_SetString (PyExc_TypeError, "error parsing input");
	    return NULL;
	}

    images_npyarray = (PyArrayObject *) PyArray_FROM_O (images_obj);
    if (PyArray_NDIM (images_npyarray) != 3)
	{
	    PyErr_SetString (PyExc_IndexError, "expected ndim=3 images array");
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}
    if (PyArray_TYPE (images_npyarray) != NPY_UINT16)
	{
	    PyErr_SetString (PyExc_RuntimeError, "expected uint16 images array");
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}

    mask_npyarray = (PyArrayObject *) PyArray_FROM_O (mask_obj);
    if (PyArray_NDIM (mask_npyarray) != 2)
	{
	    PyErr_SetString (PyExc_IndexError, "expected ndim=1 mask array");
	    Py_XDECREF (mask_npyarray);
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}
    if (PyArray_TYPE (mask_npyarray) != NPY_UINT16)
	{
	    PyErr_SetString (PyExc_RuntimeError, "expected uint16 mask array");
	    Py_XDECREF (mask_npyarray);
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}

    images_shape = PyArray_SHAPE (images_npyarray);
    mask_shape = PyArray_SHAPE (mask_npyarray);
    if (images_shape[1] != mask_shape[0] || images_shape[2] != mask_shape[1])
	{
	    PyErr_SetString (PyExc_IndexError, "mask and image sizes do not match");
	    Py_XDECREF (mask_npyarray);
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}

    /* allocate memory for return array */
    npy_intp n_bins = 65536;
    histogram_obj = PyArray_Zeros (1, &n_bins, PyArray_DescrFromType (NPY_UINT64), 0);
    histogram_npyarray = (PyArrayObject *) histogram_obj;

    /* get pointers that simulate C-style array for easier iteration */
    if (PyArray_AsCArray
	((PyObject **) & images_npyarray, (void *) &images, images_shape, 3, PyArray_DescrFromType (NPY_UINT16)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError, "conversion of images to c array failed");
	    Py_XDECREF (mask_npyarray);
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}
    if (PyArray_AsCArray
	((PyObject **) & mask_npyarray, (void *) &mask, mask_shape, 2, PyArray_DescrFromType (NPY_UINT16)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError, "conversion of mask to c array failed");
	    PyArray_Free (images_obj, images);
	    Py_XDECREF (mask_npyarray);
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}

    if (PyArray_AsCArray
	((PyObject **) & histogram_npyarray, (void *) &histogram, &n_bins, 1, PyArray_DescrFromType (NPY_UINT64)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError, "conversion of histogram to c array failed");
	    PyArray_Free (mask_obj, mask);
	    PyArray_Free (images_obj, images);
	    Py_XDECREF (mask_npyarray);
	    Py_XDECREF (images_npyarray);
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
			    if (mask[j][k] == 1)
				{
				    histogram[images[i][j][k]]++;
				}
			}
		}
	}

    Py_END_ALLOW_THREADS;

    /* keep track of the reference counting to make sure the python garbage collection can do it's thing */
    PyArray_Free (images_obj, images);
    PyArray_Free (mask_obj, mask);
    PyArray_Free (histogram_obj, histogram);
    Py_DECREF (images_npyarray);
    Py_DECREF (mask_npyarray);

    return histogram_obj;
}


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
	    PyErr_SetString (PyExc_IndexError, "expected ndim=3 images array");
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}
    if (PyArray_TYPE (images_npyarray) != NPY_UINT16)
	{
	    PyErr_SetString (PyExc_RuntimeError, "expected uint16 images array");
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}

    mask_npyarray = (PyArrayObject *) PyArray_FROM_O (mask_obj);
    if (PyArray_NDIM (mask_npyarray) != 2)
	{
	    PyErr_SetString (PyExc_IndexError, "expected ndim=1 mask array");
	    Py_XDECREF (mask_npyarray);
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}
    if (PyArray_TYPE (mask_npyarray) != NPY_UINT16)
	{
	    PyErr_SetString (PyExc_RuntimeError, "expected uint16 mask array");
	    Py_XDECREF (mask_npyarray);
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}

    images_shape = PyArray_SHAPE (images_npyarray);
    mask_shape = PyArray_SHAPE (mask_npyarray);
    if (images_shape[1] != mask_shape[0] || images_shape[2] != mask_shape[1])
	{
	    PyErr_SetString (PyExc_IndexError, "mask and image sizes do not match");
	    Py_XDECREF (mask_npyarray);
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}

    /* allocate memory for return array */
    sum_obj = PyArray_Zeros (1, &images_shape[0], PyArray_DescrFromType (NPY_UINT64), 0);
    sum_npyarray = (PyArrayObject *) sum_obj;

    /* get pointers that simulate C-style array for easier iteration */
    if (PyArray_AsCArray
	((PyObject **) & images_npyarray, (void *) &images, images_shape, 3, PyArray_DescrFromType (NPY_UINT16)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError, "conversion of images to c array failed");
	    Py_XDECREF (mask_npyarray);
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}
    if (PyArray_AsCArray
	((PyObject **) & mask_npyarray, (void *) &mask, mask_shape, 2, PyArray_DescrFromType (NPY_UINT16)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError, "conversion of mask to c array failed");
	    PyArray_Free (images_obj, images);
	    Py_XDECREF (mask_npyarray);
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}

    npy_intp *n_imgs[1];
    n_imgs[0] = &images_shape[0];
    if (PyArray_AsCArray
	((PyObject **) & sum_npyarray, (void *) &sum, *n_imgs, 1, PyArray_DescrFromType (NPY_UINT64)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError, "conversion of sums to c array failed");
	    PyArray_Free (mask_obj, mask);
	    PyArray_Free (images_obj, images);
	    Py_XDECREF (mask_npyarray);
	    Py_XDECREF (images_npyarray);
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
    PyArray_Free (images_obj, images);
    PyArray_Free (mask_obj, mask);
    PyArray_Free (sum_obj, sum);
    Py_DECREF (images_npyarray);
    Py_DECREF (mask_npyarray);

    return sum_obj;
}


static PyObject *
normed_sum (PyObject * self, PyObject * args)
{
    PyObject *sum_img_obj;
    PyArrayObject *sum_img_npyarray;
    PyObject *images_obj, *norm_values_obj;
    PyArrayObject *images_npyarray, *norm_values_npyarray;
    npy_intp *images_shape;
    npy_intp *norm_values_shape;
    npy_uint16 ***images;
    npy_float32 **sum_img, *norm_values;

    /* parse input objects, check dimensions and data types */
    if (!PyArg_ParseTuple (args, "OO", &images_obj, &norm_values_obj))
	{
	    PyErr_SetString (PyExc_TypeError, "error parsing input");
	    return NULL;
	}

    images_npyarray = (PyArrayObject *) PyArray_FROM_O (images_obj);
    if (PyArray_NDIM (images_npyarray) != 3)
	{
	    PyErr_SetString (PyExc_IndexError, "expected ndim=3 images array");
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}
    if (PyArray_TYPE (images_npyarray) != NPY_UINT16)
	{
	    PyErr_SetString (PyExc_RuntimeError, "expected uint16 images array");
	    Py_XDECREF (images_npyarray);
	    return NULL;
	}

    norm_values_npyarray = (PyArrayObject *) PyArray_FROM_O (norm_values_obj);
    if (PyArray_NDIM (norm_values_npyarray) != 1)
	{
	    PyErr_SetString (PyExc_IndexError, "expected ndim=1 norm_values array");
	    Py_XDECREF (images_npyarray);
	    Py_XDECREF (norm_values_npyarray);
	    return NULL;
	}
    if (PyArray_TYPE (norm_values_npyarray) != NPY_FLOAT32)
	{
	    PyErr_SetString (PyExc_RuntimeError, "expected float32 norm_values array");
	    Py_XDECREF (images_npyarray);
	    Py_XDECREF (norm_values_npyarray);
	    return NULL;
	}

    images_shape = PyArray_SHAPE (images_npyarray);
    norm_values_shape = PyArray_SHAPE (norm_values_npyarray);
    if (images_shape[0] != norm_values_shape[0])
	{
	    PyErr_SetString (PyExc_IndexError, "normed_values and image sizes do not match");
	    Py_XDECREF (images_npyarray);
	    Py_XDECREF (norm_values_npyarray);
	    return NULL;
	}

    /* allocate memory for return array */
    npy_intp *image_shape[2] = { (npy_intp *) images_shape[1], (npy_intp *) images_shape[2] };
    sum_img_obj = PyArray_Zeros (2, (npy_intp *) image_shape, PyArray_DescrFromType (NPY_FLOAT32), 0);
    sum_img_npyarray = (PyArrayObject *) sum_img_obj;

    /* get pointers that simulate C-style array for easier iteration */
    if (PyArray_AsCArray
	((PyObject **) & images_npyarray, (void *) &images, images_shape, 3, PyArray_DescrFromType (NPY_UINT16)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError, "conversion of images to c array failed");
	    Py_XDECREF (images_npyarray);
	    Py_XDECREF (norm_values_npyarray);
	    return NULL;
	}
    if (PyArray_AsCArray
	((PyObject **) & norm_values_npyarray, (void *) &norm_values,
	 norm_values_shape, 1, PyArray_DescrFromType (NPY_FLOAT32)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError, "conversion of norm_values to c array failed");
	    PyArray_Free (images_obj, (void *) images);
	    Py_XDECREF (images_npyarray);
	    Py_XDECREF (norm_values_npyarray);
	    return NULL;
	}

    if (PyArray_AsCArray
	((PyObject **) & sum_img_npyarray, (void *) &sum_img,
	 (npy_intp *) image_shape, 2, PyArray_DescrFromType (NPY_FLOAT32)) == -1)
	{
	    PyErr_SetString (PyExc_RuntimeError, "conversion of sum_img to c array failed");
	    PyArray_Free (images_obj, (void *) images);
	    PyArray_Free (norm_values_obj, (void *) norm_values);
	    Py_XDECREF (images_npyarray);
	    Py_XDECREF (norm_values_npyarray);
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
			    sum_img[j][k] += images[i][j][k] / norm_values[i];
			}
		}
	}

    Py_END_ALLOW_THREADS;

    /* keep track of the reference counting to make sure the python garbage collection can do it's thing */
    PyArray_Free (images_obj, images);
    PyArray_Free (norm_values_obj, norm_values);
    PyArray_Free (sum_img_obj, sum_img);
    Py_DECREF (images_npyarray);
    Py_DECREF (norm_values_npyarray);
    return sum_img_obj;
//    return Py_None;
}
