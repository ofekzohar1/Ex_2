#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define SQ(x) ((x)*(x))
/* Custom made python type error */
#define MyPy_TypeErr(x, y) \
PyErr_Format(PyExc_TypeError, "%s type is required (got type %s)", x ,Py_TYPE(y)->tp_name) \

typedef struct {
    double *prevCentroid;
    double *currCentroid;
    int counter; /* Number of vectors (datapoints) in cluster */
} Cluster;

int initVectorsArray(double ***vectorsArrayPtr, const int *numOfVectors, const int *dimension, PyObject *pyVectorsList); /* Insert vectors into an array */
int initClusters(Cluster **clustersArrayPtr, double **vectorsArray, const int *k, const int *dimension, const int *firstCentralIndexes); /* Initialize empty clusters array */
double vectorsNorm(const double *vec1, const double *vec2, const int *dimension); /* Calculate the norm between 2 vectors */
int findMyCluster(double *vec, Cluster *clustersArray, const int *k, const int *dimension); /* Return the vector's closest cluster (in terms of norm) */
void assignVectorsToClusters(double **vectorsArray, Cluster *clustersArray, const int *k, const int *numOfVectors, const int *dimension); /* For any vector assign to his closest cluster */
int recalcCentroids(Cluster *clustersArray, const int *k, const int *dimension); /* Recalculate clusters' centroids, return number of changes */
void initCurrCentroidAndCounter(Cluster *clustersArray, const int *k, const int *dimension); /* Set curr centroid to prev centroid and reset the counter */
PyObject *buildPyListCentroids(Cluster *clustersArray, const int *k, const int *dimension); /* Print clusters' final centroids */
void freeMemoryVectorsClusters(double **vectorsArray, Cluster *clustersArray, const int *k, int *firstCentralIndexes); /* Free the allocated memory */

static PyObject* fit(int k, int maxIter, int dimension, int numOfVectors, double **vectorsArray, int* firstCentralIndexes, Cluster **clustersArrayPtr) {
    int i, changes;

    /* Initialize clusters arrays */
    if (initClusters(clustersArrayPtr, vectorsArray, &k, &dimension, firstCentralIndexes)) /* 0 (false) on success, true on malloc fail */
        return PyErr_NoMemory();

    for (i = 0; i < maxIter; ++i) {
        initCurrCentroidAndCounter(*clustersArrayPtr, &k, &dimension); /* Update curr centroid to prev centroid and reset the counter */
        assignVectorsToClusters(vectorsArray, *clustersArrayPtr, &k, &numOfVectors, &dimension);
        changes = recalcCentroids(*clustersArrayPtr, &k, &dimension); /* Calculate new centroids */
        if (changes == 0) { /* Centroids stay unchanged in the current iteration */
            break;
        }
    }

    return buildPyListCentroids(*clustersArrayPtr, &k, &dimension);
}

/*
 * This actually defines the fit function using a wrapper C API function
 * The wrapping function needs a PyObject* self argument.
 * This is a requirement for all functions and methods in the C API.
 * It has input PyObject *args from Python.
 */
static PyObject* fit_connect(PyObject *self, PyObject *args) {
    Py_ssize_t i;
    PyObject *pyCentralsList, *pyVectorsList, *resault = NULL;
    int k, maxIter, dimension, numOfVectors, *firstCentralIndexes;
    double **vectorsArray = NULL; /* Default value - not allocated */
    Cluster *clustersArray = NULL; /* Default value - not allocated */

    if (!PyArg_ParseTuple(args, "iiiiOO", &k, &maxIter, &dimension, &numOfVectors, &pyCentralsList, &pyVectorsList))
        return NULL; /* Type error - not in correct format */
    if (!PyList_Check(pyCentralsList)) {
        MyPy_TypeErr("List", pyCentralsList);
        return NULL; /* Type error - not a python List */
    }
    if (!PyList_Check(pyVectorsList))
    {
        MyPy_TypeErr("List", pyVectorsList);
        return NULL; /* Type error - not a python List */
    }

    firstCentralIndexes = (int *) malloc(k * sizeof(int));
    if (firstCentralIndexes == NULL) {
        resault = PyErr_NoMemory(); /* Memory allocation error */
        goto end;
    }
    for (i = 0; i < k; i++) {
        firstCentralIndexes[i] = (int)PyLong_AsLong(PyList_GetItem(pyCentralsList, i));
        if (PyErr_Occurred()) {
            goto end; /* Casting error to int */
        }
    }
    if (initVectorsArray(&vectorsArray, &numOfVectors, &dimension, pyVectorsList)) /* return 0 (false) on success, true on error */
        goto end; /* On any error from initVectorsArray() */
    resault = fit(k, maxIter, dimension, numOfVectors, vectorsArray, firstCentralIndexes, &clustersArray);

    end:
    freeMemoryVectorsClusters(vectorsArray, clustersArray, &k, firstCentralIndexes); /* Free memory */
    return resault;
}

/*
 * This array tells Python what methods this module has.
 * We will use it in the next structure
 */
static PyMethodDef _method[] = {
        {"fit",                      /* the Python method name that will be used */
                (PyCFunction) fit_connect, /* the C-function that implements the Python function and returns static PyObject*  */
                METH_VARARGS,   /* flags indicating parametersaccepted for this function */
                        NULL},      /*  The docstring for the function (PyDoc_STR("")) */
        {NULL, NULL, 0, NULL}        /* The is a sentinel. Python looks for this entry to know that all
                                       of the functions for the module have been defined. */
};

/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp", /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
        _method /* the PyMethodDef array from before containing the methods of the extension */
};

/*
 * The PyModuleDef structure, in turn, must be passed to the interpreter in the module’s initialization function.
 * The initialization function must be named PyInit_name(), where name is the name of the module and should match
 * what we wrote in struct PyModuleDef.
 * This should be the only non-static item defined in the module file
 */
PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    return PyModule_Create(&moduledef);
}

int initVectorsArray(double ***vectorsArrayPtr, const int *numOfVectors, const int *dimension, PyObject *pyVectorsList) {
    Py_ssize_t i, j;
    double *matrix;
    PyObject *vector, *comp;
    /* Allocate memory for vectorsArrayPtr */
    matrix = (double *) malloc((*numOfVectors) * ((*dimension) + 1) * sizeof(double));
    *vectorsArrayPtr = malloc((*numOfVectors) * sizeof(double *));
    if (matrix == NULL || (*vectorsArrayPtr) == NULL){
        if (matrix != NULL)
            free(matrix); /* Free matrix if exist */
        PyErr_SetNone(PyExc_MemoryError);
        return 1; /* Memory allocation error */
    }

    for (i = 0; i < *numOfVectors; ++i) {
        (*vectorsArrayPtr)[i] = matrix + i * ((*dimension) + 1); /* Set VectorsArray to point to 2nd dimension array */
        vector = PyList_GetItem(pyVectorsList, i);
        if (!PyList_Check(vector)) {
            MyPy_TypeErr("List", vector);
            free(*vectorsArrayPtr);
            free(matrix);
            *vectorsArrayPtr = NULL;
            return 1; /* Type error - not a python List */
        }
        for (j = 0; j < *dimension; ++j) {
            comp = PyList_GetItem(vector, j);
            (*vectorsArrayPtr)[i][j] = PyFloat_AsDouble(comp);
            if (PyErr_Occurred()) {
                free(*vectorsArrayPtr);
                free(matrix);
                *vectorsArrayPtr = NULL;
                return 1; /* Cast error to double */
            }
        }
    }
    return 0; /* Success */
}

int initClusters(Cluster **clustersArrayPtr, double **vectorsArray, const int *k, const int *dimension, const int *firstCentralIndexes) {
    int i, j, mallocFails = 0;
    /* Allocate memory for clustersArrayPtr */
    *clustersArrayPtr = (Cluster *) malloc((*k) * sizeof(Cluster));
    if ((*clustersArrayPtr) != NULL) {
        for (i = 0; i < *k; ++i) {
            (*clustersArrayPtr)[i].counter = 0;
            (*clustersArrayPtr)[i].prevCentroid = (double *) malloc((*dimension) * sizeof(double));
            (*clustersArrayPtr)[i].currCentroid = (double *) malloc((*dimension) * sizeof(double));
            if ((*clustersArrayPtr)[i].prevCentroid == NULL || (*clustersArrayPtr)[i].currCentroid == NULL) {
                mallocFails++;
                continue;
            }

            /* Assign the initial k vectors to their corresponding clusters according to the ones calculated in python */
            for (j = 0; j < *dimension && mallocFails == 0; ++j) {
                (*clustersArrayPtr)[i].currCentroid[j] = vectorsArray[firstCentralIndexes[i]][j];
            }
        }
    } else {
        mallocFails++;
    }
    return mallocFails; /* 0 on success, greater than 0 on fail */
}

double vectorsNorm(const double *vec1, const double *vec2, const int *dimension) {
    double norm = 0;
    int i;
    for (i = 0; i < *dimension; ++i) {
        norm += SQ(vec1[i] - vec2[i]);
    }
    return norm;
}

int findMyCluster(double *vec, Cluster *clustersArray, const int *k, const int *dimension) {
    int myCluster, j;
    double minNorm, norm;

    myCluster = 0;
    minNorm = vectorsNorm(vec, clustersArray[0].prevCentroid, dimension);
    for (j = 1; j < *k; ++j) { /* Find the min norm == closest cluster */
        norm = vectorsNorm(vec, clustersArray[j].prevCentroid, dimension);
        if (norm < minNorm) {
            myCluster = j;
            minNorm = norm;
        }
    }
    return myCluster;
}

void assignVectorsToClusters(double **vectorsArray, Cluster *clustersArray, const int *k, const int *numOfVectors, const int *dimension) {
    int j, i, myCluster;
    double *vec;
    for (j = 0; j < *numOfVectors; ++j) {
        vec = vectorsArray[j];
        myCluster = findMyCluster(vectorsArray[j], clustersArray, k, dimension);
        vec[*dimension] = myCluster; /* Set vector's cluster to his closest */
        for (i = 0; i < *dimension; ++i) {
            clustersArray[myCluster].currCentroid[i] += vec[i]; /* Summation of the vectors Components */
        }
        clustersArray[myCluster].counter++; /* Count the number of vectors for each cluster */
    }
}

int recalcCentroids(Cluster *clustersArray, const int *k, const int *dimension) {
    int i, j, changes = 0;
    Cluster cluster;
    for (i = 0; i < *k; ++i) {
        cluster = clustersArray[i];
        for (j = 0; j < *dimension; ++j) {
            cluster.currCentroid[j] /= cluster.counter; /* Calc the mean value */
            changes += cluster.prevCentroid[j] != cluster.currCentroid[j] ? 1 : 0; /* Count the number of changed centroids' components */
        }
    }
    return changes;
}

void initCurrCentroidAndCounter(Cluster *clustersArray, const int *k, const int *dimension) {
    int i, j;
    for (i = 0; i < *k; ++i) {
        for (j = 0; j < *dimension; ++j) {
            clustersArray[i].prevCentroid[j] = clustersArray[i].currCentroid[j]; /* Set prev centroid to curr centroid */
            clustersArray[i].currCentroid[j] = 0; /* Reset curr centroid */
        }
        clustersArray[i].counter = 0; /* Reset counter */
    }
}

PyObject *buildPyListCentroids(Cluster *clustersArray, const int *k, const int *dimension) {
    Py_ssize_t i, j;
    PyObject *listOfCentrals, *central, *comp;
    listOfCentrals = PyList_New(*k);
    if (listOfCentrals != NULL) { /* If NULL alloc fail */
        for (i = 0; i < *k; ++i) {
            central = PyList_New(*dimension);
            if (central == NULL) {
                Py_DecRef(listOfCentrals);
                return NULL; /* If NULL alloc fail */
            }
            for (j = 0; j < *dimension; ++j) {
                comp = PyFloat_FromDouble(clustersArray[i].currCentroid[j]);
                if (comp == NULL || PyList_SetItem(central, j, comp)) {
                    Py_DecRef(listOfCentrals);
                    Py_DecRef(central);
                    Py_XDECREF(comp);
                    return NULL; /* Set error */
                }
            }
            if (PyList_SetItem(listOfCentrals, i, central)) {
                Py_DecRef(listOfCentrals);
                Py_DecRef(central);
                return NULL; /* Set error */
            }
        }
    }
    return listOfCentrals;
}

void freeMemoryVectorsClusters(double **vectorsArray, Cluster *clustersArray, const int *k, int *firstCentralIndexes) {
    int i;
    /* Free clusters */
    if (clustersArray != NULL) {
        for (i = 0; i < *k; ++i) {
            free(clustersArray[i].currCentroid);
            free(clustersArray[i].prevCentroid);
        }
    }
    free(clustersArray);
    free(firstCentralIndexes);

    /* Free vectors */
    if (vectorsArray != NULL)
        free(*vectorsArray);
    free(vectorsArray);
}