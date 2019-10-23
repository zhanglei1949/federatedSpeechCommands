import torch
import numpy as np
def getLenOfGradientVector(current_grad):
    #expect a list consists of numpy arrays
    n = 0
    for arr in current_grad:
        #print(list(arr.view(-1,).shape)[0])
        n += list(arr.view(-1,).shape)[0]
    return n
def getShapeList(current_grad):
    # return the list of shapes of grad vectors, for recover
    res = []
    for arr in current_grad:
        res.append(arr.shape)
    return res
def transListOfArraysToArrays(current_grad, n):
    # expect a list of arrays, return a squeezed array, n is the total length
    res = np.zeros((n))
    ind = 0
    for arr in current_grad:
        arr = arr.view(-1,).cpu()
        res[ind:ind+arr.shape[0]] = arr
        ind+=arr.shape[0]
    return res
def listMulti(l):
    res = 1
    for ele in l:
        res *= ele
    return res
def transNumpyGrad2Cuda(grad, shape_list):
    res = []
    ind = 0
    for shape in shape_list:
        tmp = torch.from_numpy(grad[ind:ind+listMulti(shape)])
        tmp = tmp.view(shape)
        res.append(tmp.float().cuda())
        ind += listMulti(shape)
    return res