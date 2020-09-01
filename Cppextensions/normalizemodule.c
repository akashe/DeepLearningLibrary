#include <Python.h>

static PyObject *
normalize_normalize(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;

    char z[100] = "Holy shit its really working";

    printf("%s", z);
    //sts = system(command);
    //return Py_BuildValue("i", sts);
    return 0;
}