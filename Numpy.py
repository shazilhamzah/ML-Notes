import numpy as np
import timeit

def createArray():
    print("\nExecuting createArray()")
    npArray = np.arange(1,10)
    print(npArray)
    print("Done executing createArray()")
    
def createArrayFromInput():
    print("\nExecuting createArrayFromInput()")
    l = []
    for i in range(0,5):
        l.append(int(input("Enter a number: ")))
    npArray = np.array(l);
    print(npArray)
    print("Done executing createArrayFromInput()")

def checkDimension():
    print("\nExecuting checkDimension()")
    npArray = np.arange(1,10)
    print("Array dimension: ",npArray.ndim)
    print("Done executing checkDimension()")
    
def twoDimmensionalArray():
    print("\nExecuting twoDimmensionalArray()")
    npArray = np.array([[1,2,3,4],[5,6,7,8]])
    print(npArray)
    print("Array dimension: ",npArray.ndim)
    print("Done executing twoDimmensionalArray()")
    
def threeDimmensionalArray():
    print("\nExecuting threeDimmensionalArray()")
    npArray = np.array([[[1,2,3,4],[1,2,3,4]],[[5,6,7,8],[5,6,7,8]]])
    print(npArray)
    print("Array dimension: ",npArray.ndim)
    print("Done executing threeDimmensionalArray()")
    
def nDimmensionalArray():
    print("\nExecuting nDimmensionalArray()")
    npArray = np.array([1,2,3,4],ndmin=10)
    print(npArray)
    print("Array dimension: ",npArray.ndim)
    print("Done executing nDimmensionalArray()")
    
def zeroArray():
    print("\nExecuting zeroArray()")
    npArray = np.zeros((3,4))
    print(npArray)
    print("Done executing zeroArray()")
    
def oneArray():
    print("\nExecuting oneArray()")
    npArray = np.ones((3,4))
    print(npArray)
    print("Done executing oneArray()")
    
def identityMatrix():
    print("\nExecuting identityMatrix()")
    npArray = np.eye(3)
    npArray2 = np.eye(3,5)
    print(npArray)
    print()
    print(npArray2)
    print("Done executing identityMatrix()")
    
def linearGapArray():
    print("\nExecuting linearGapArray()")
    npArray = np.linspace(0,20,num=7)   # PRINT 6 NUMBERS FROM 0 TO 20
    print(npArray)   
    print("Done executing linearGapArray()")

def arrayWithRandValues():
    print("\nExecuting arrayWithRandValues()")
    npArray = np.random.rand(2,3)         # GENERATE ARRAY OF 4 ELEMENTS WITH RANDOM NUMBERS BETWEEN 0 AND 1
    npArray1 = np.random.randn(2)         # RANDOM NUMBERS CLOSE TO 0
    npArray2 = np.random.ranf(2)          # RANDOM NUMBERS IN RANGE [0,1) INCLUDES 0 BUT NOT 1
    npArray3 = np.random.randint(0,10,5)  # 5 RANDOM NUMBERS IN RANGE (0,10)
    print(npArray)   
    print(npArray1)   
    print(npArray2)   
    print(npArray3)
    print("Done executing arrayWithRandValues()")
    
def dataTypeConv():
    print("\nExecuting dataTypeConv()")
    npArray = np.array([1,2,3,4],dtype="f")
    npArray1 = np.bool_(npArray)
    npArray2 = npArray.astype(float)
    print(npArray)
    print("Array Type: ",npArray.dtype)
    print(npArray1)
    print("Array1 Type: ",npArray1.dtype)
    print(npArray2)
    print("Array1 Type: ",npArray2.dtype)
    print("Done executing dataTypeConv()")

    
def arrayArithematic():
    print("\nExecuting arrayArithematic()")
    npArray = np.array([1,2,3,4])
    npArray1 = np.array([1,2,3,4])
    npArray2 = np.array([1,2,3,4])
    print(npArray+3)
    print(npArray1*npArray2)    # YOU CAN ALSO USE FUNCTIONS LIKE np.add(arr1,arr2) etc.
    print("Done executing arrayArithematic()")
    
def arrayFunctions():
    print("\nExecuting arrayFunctions()")
    npArray = np.array([1,2,3,4])
    print("Min: ",np.min(npArray),np.argmin(npArray))   
    print("Max: ",np.max(npArray),np.argmax(npArray))
    print("Sqrt: ",np.sqrt(npArray))
    print("Sin: ",np.sin(npArray))
    print("Cos: ",np.cos(npArray))
    print("Cumsum: ",np.cumsum(npArray))        # CUMMULATIVE SUM
    npArray1 = np.array([[1,2,3,4],[5,6,7,8]])
    print("Min in 2D",np.min(npArray1,axis=0))  # AXIS 1 IS FOR ROW AND 0 IS FOR COLUMN
    print("Done executing arrayFunctions()")
    
def shape():    # SHAPE ALSO MEANS DIMENSIONS
    print("\nExecuting shape()")
    npArray = np.array([1,2,3,4])
    npArray1 = np.array([[1,2,3,4],[1,2,3,4]])
    npArray2 = np.array([[[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4]]])
    print("Shape of 1D Array: ",np.shape(npArray))
    print("Shape of 2D Array: ",np.shape(npArray1))     # 2 ROWS AND 4 COLUMNS
    print("Shape of 3D Array: ",np.shape(npArray2))     # 2 ROWS IN ONE LAYER AND 2 ROWS IN OTHER LAYER AND 4 COLUMNS
    print((npArray2))
    print("Done executing shape()")
    
def reshape():    # SHAPE ALSO MEANS DIMENSIONS
    print("\nExecuting reshape()")
    npArray = np.array([1,2,3,4])
    renpArray = npArray.reshape(2,2)        # THERE MUST NOT BE AN EMPTY ARRAY AFTER DIVISION, OTHERWISE IT MAY THROW ERROR
    print(renpArray)
    print("Done executing reshape()")
    
# BROADCASTING ERROR
'''
    BROADCASTING ERROR OCCURS WHEN YOU TRY TO PERFORM OPERATION ON ARRAYS OF DIFFERENT DIMENSIONS.
    IT DOESN'T OCCUR WHEN YOU ARE PERFORMING OPERATION ON 1D ARRAYS (ROW OR COLUMN MATRIX).
    ADDITION OF ROW AND COLUMN MATRIX GIVES A 3X3 MATRIX.
    STARTING FROM RIGHT SIDE, IF ANY OF THE DIMENSION IS 1, WE CAN APPLY ARITHEMATIC OPERATION.
'''

def slicing():
    print("\nExecuting slicing()")
    npArray = np.array([1,2,3,4])
    print("Index 0 to 2: ",npArray[:3])
    print("Index 1 to 3: ",npArray[1:])
    print("Skipping 2: ",npArray[::2])
    npArray1 = np.array([[1,2,3,4],[5,6,7,8]])
    print("Index 0 to 2 of 2nd row of 2D array: ",npArray1[1,:3])
    """
        FOR n DIMENSIONS, USE COMMAS TO REACH A MINIMAL ROW AND THEN USE COLUMNS METHOD.
        FOR EXAMPLE, IN 4D ARRAY, use [1,1,1,1:3:2]
    """
    print("Done executing slicing()")
    
def iteration():
    print("\nExecuting iteration()")
    npArray = np.array([1,2,3,4])
    npArray1 = np.array([[1,2,3,4],[5,6,7,8]])
    print("PRINTING 2D ARRAY: ",end=' ')
    for i in np.nditer(npArray1):
        print(i,sep='',end=' ')
    print()
    print("PRINTING 2D ARRAY AND SAVING PRINTED IN BUFFER: ",end=' ')
    for i in np.nditer(npArray1,flags=['buffered'],op_dtypes=["S"]):    # THESE ARE RESERVED KEYWORDS, S MEANS STRING
        print(i,end=' ')
    print()
    print("PRINTING 2D ARRAY AND ALONG WITH THE INDEXES: ",end=' ')
    for i,d in np.ndenumerate(npArray1):    # THESE ARE RESERVED KEYWORDS, S MEANS STRING
        print(i,d)
    print()
    print("Done executing iteration()")
    
# copy() VS view()
"""
    copy() IS DEEP COPY i.e. IT CREATES A NEW ARRAY.
    view() IS SHALLOW COPY i.e. IT DOESN'T CREATE A NEW ARRAY, IT JUST POINTS TOWARDS THE OLD ARRAY.
"""

def join():
    print("\nExecuting join()")
    npArray1 = np.array([[1,2,3,4],[5,6,7,8]])
    npArray2 = np.array([[9,10,11,12],[13,14,15,16]])
    print("JOINING HORIZONTALLY: ",np.hstack((npArray1,npArray2)))
    print("JOINING VERTICALLY: ",np.vstack((npArray1,npArray2)))
    print("JOINING DEPTH WISE: ",np.dstack((npArray1,npArray2)))
    print("Done executing join()")
    
def split():
    print("\nExecuting split()")
    npArray1 = np.array([[1,2,3,4],[5,6,7,8]])
    print("Splitting horizontally: ",np.array_split(npArray1,4,axis=0)) # SPLIT IN 8 PARTS
    print("Done executing split()")
    
# MISC. FUNCTIONS
"""
    np.searchsorted(arrayName,value(s)) -> RETURN INDEX/INDICES WHERE IT CAN BE INSERTED IN SORTED ARRAY
    np.where((arrayName%2)==0) -> RETURNS INDEXES OF EVEN NUMBERS
    np.sort(arrayName) -> SORTS ARRAY
    np.random.shuffle(arrayName)
    np.unique(arrayName,return_index=True,return_count=True)
    np.resize(arrayName,(x,y))
    arrayName.flatten() -> CONVERTS nD ARRAY INTO 1D ARRAY (DEEP COPY)
    np.ravel(arrayName) -> CONVERTS nD ARRAY INTO 1D ARRAY (SHALLOW COPY)
"""

def insert():
    print("\nExecuting insert()")
    npArray = np.array([1,2,3,4])
    npArray1 = np.array([[1,2,3,4],[5,6,7,8]])
    new_npArray = np.insert(npArray,3,40)     # INSERT 40 AT INDEX 3
    new_npArray1 = np.insert(npArray1,2,6,axis=0)   # INSERT 6 AT INDEX 2 ALONG AXIS 0 (ADD ANOTHER ROW)
    new_npArray2 = np.insert(npArray1,2,6,axis=1)   # INSERT 6 AT INDEX 2 ALONG AXIS 0 (ADD ANOTHER COLUMN)
    print(new_npArray)
    print(new_npArray1)
    print(new_npArray2)
    print("Done executing insert()")
    
# MATRIX FUNCTIONS
"""
    YOU CAN CREATE MATRIX USING np.matrix()
    FUNCTIONS:
    -   np.transpose(matrixName) OR matrixName.T
    -   np.linalg.inv(matrixName)  -> INVERSE OF MATRIX
    -   np.linalg.matrix_power(matrixName,2)    -> SQUARE OF MATRIX
    -   np.linalg.det(matrixName)  -> DETERMINANT OF MATRIX
"""

def matrix():
    print("\nExecuting matrix()")
    matrixName = np.matrix([[1,2],[3,4]])
    print("Original Matrix: ",matrixName)
    print("Transpose: ",matrixName.T)
    print("Inverse: ",np.linalg.inv(matrixName))
    print("Power: ",np.linalg.matrix_power(matrixName,2))
    print("Determinant: ",np.linalg.det(matrixName))
    print("Done executing matrix()")
    




print("Time taken by nparray",timeit.timeit(createArray,number=1),end="\n\n")
# print("Time taken by nparray",timeit.timeit(createArrayFromInput,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(checkDimension,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(twoDimmensionalArray,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(threeDimmensionalArray,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(nDimmensionalArray,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(zeroArray,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(oneArray,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(identityMatrix,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(linearGapArray,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(arrayWithRandValues,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(dataTypeConv,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(arrayArithematic,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(arrayFunctions,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(shape,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(reshape,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(slicing,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(iteration,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(join,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(split,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(insert,number=1),end="\n\n")
print("Time taken by nparray",timeit.timeit(matrix,number=1),end="\n\n")