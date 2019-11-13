import torch, threading
import time
import numpy as np
import math
import random
from scipy.linalg import null_space
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

############ gpu version
def getLenOfGradientVectorCuda(current_grad):
    #expect a list consists of numpy arrays
    n = 0
    for arr in current_grad:
        #print(list(arr.view(-1,).shape)[0])
        n += list(arr.view(-1,).shape)[0]
    return n
def getShapeListCuda(current_grad):
    # return the list of shapes of grad vectors, for recover
    res = []
    for arr in current_grad:
        res.append(arr.shape)
    return res
def get_shape_and_length_gradient_cpu(current_grad):
    n = 0
    res = []
    for arr in current_grad:
        n += list(arr.view(-1,).shape)[0]
        res.append(arr.shape)
    return n, res
def transListOfArraysToArraysCpu(current_grad, n):
    # expect a list of arrays, return a squeezed cuda array, n is the total length
    res = np.zeros((n))
    ind = 0
    for arr in current_grad:
        arr = arr.view(-1,)
        res[ind:ind+arr.shape[0]] = arr.cpu().numpy()
        ind+=arr.shape[0]
    return res
def trans2numpyArrayWithShapeList(grad, shape_list):
    res = []
    ind = 0
    for shape in shape_list:
        tmp = grad[ind:ind+listMulti(shape)]
        tmp = tmp.view(shape)
        res.append(tmp)
        ind += listMulti(shape)
    return res

class Federated:
    def __init__(self, num_clients, matrix_size):
        # utilize a sample gradient vector 
        self.num_clients = num_clients
        self.matrix_size = matrix_size
        self.num_threads = 10
        self.MAX = 0.001
        self.S_i = random.sample(range(0, 3 * self.matrix_size), self.matrix_size)
        self.S_i.sort()
        self.S_j = random.sample(range(0, 2 * self.matrix_size), self.matrix_size)
        self.S_j.sort()
    def init(self, gradient):
        self.len_gradient, self.shape_list = get_shape_and_length_gradient_cpu(gradient)
        self.len_gradient_after_padding = 3 * math.ceil(float(self.len_gradient)/self.matrix_size) * self.matrix_size
        
        self.ori_gradient_sum = np.zeros((self.len_gradient))
        self.random_gradient_sum = np.zeros((self.len_gradient_after_padding * 3))
        
        self.A = self.MAX * np.random.rand(self.matrix_size, self.matrix_size)
        self.A_inv = np.linalg.inv(self.A)
        self.B = np.zeros((self.matrix_size, 3 * self.matrix_size))
        for i in range(0, self.matrix_size):
            self.B[:, self.S_i[i] : self.S_i[i]+1] = self.A[:, i:i+1]
        self.C = np.random.rand(2 * self.matrix_size, 3 * self.matrix_size) * self.MAX
        for i in range(0, self.matrix_size):
            self.C[self.S_j[i] : self.S_j[i] + 1, :] = self.B[i:i+1 , :]
        
        # SVD
        self.u, self.s, self.vh = np.linalg.svd(self.C, full_matrices = True)
        self.vh_t = np.transpose(self.vh)
        self.sigma = np.zeros((self.C.shape[0], self.C.shape[1]))
        self.sigma[: self.s.shape[0], : self.s.shape[0]] = np.diag(self.s)
        # null space
        self.u_sigma = np.dot(self.u, self.sigma)
        self.ns = null_space(self.C) # (3000, 1000) we use the first args.
        print("Initializatin complete", self.ns.shape)
    def work_for_client(self, client_no, gradient):
        time1 = time.time()
        assert(client_no < self.num_clients)
        flatterned_grad = transListOfArraysToArraysCpu(gradient, self.len_gradient)
        self.ori_gradient_sum += flatterned_grad
        # padding
        flatterned_grad_extended = np.zeros((self.len_gradient_after_padding, 1))
        flatterned_grad_extended[:self.len_gradient, 0] = flatterned_grad
        kernel_space = np.zeros((3 * self.matrix_size, 1))
        random_numbers = self.MAX * np.random.rand(self.matrix_size, 1)

        for i in range(self.matrix_size):
            kernel_space += random_numbers[i] * self.ns[:, i:i+1]
        
        flatterned_grad_extended_after_random = self.MAX * np.random.rand(3 * self.len_gradient_after_padding, 1)

        for i in range(0, self.len_gradient_after_padding, self.matrix_size):
            for j in range(0, self.matrix_size):
                flatterned_grad_extended_after_random[i * 3 + self.S_i[j]] = flatterned_grad_extended[i + j]
        
        # compute result
        time2 = time.time()
        print("client ", client_no, " randomization complete")
        flatterned_grad_extended_final = self.MAX * np.random.rand(3 * self.len_gradient_after_padding, 1)
        ##TODO: optimize matrix calculation
        '''
        for i in range(0, self.len_gradient_after_padding, self.matrix_size):
            for j in range(0, self.matrix_size):
                flatterned_grad_extended_final[3 * i : 3*(i + self.matrix_size), :] = np.dot(np.transpose(self.vh), flatterned_grad_extended_after_random[3 * i : 3*(i + self.matrix_size), :] + kernel_space)
        '''
        def matrixProd(thread_id):
            part_num = self.len_gradient_after_padding / 10
            print(threading.current_thread().name, thread_id * part_num, (thread_id + 1) * part_num)
            
            for i in range(int(thread_id * part_num), int((thread_id + 1) * part_num), self.matrix_size):
                for j in range(0, self.matrix_size): 
                    flatterned_grad_extended_final[3 * i : 3*(i + self.matrix_size), :] \
                    = np.dot(self.vh_t, flatterned_grad_extended_after_random[3 * i : 3*(i + self.matrix_size), :] + kernel_space)
            print(threading.current_thread().name, " finish")
        threads = []
        for _i in range(self.num_threads):
            t = threading.Thread(target = matrixProd, args = (_i,))
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()
        self.random_gradient_sum += flatterned_grad_extended_final[:, 0]
        time3 = time.time()
        print("client ",  client_no, " masking complete",)
        print("time for randomization ", time2 - time1, "time for masking", time3 - time2)
    def recoverGradient(self):
        time1 = time.time()
        res = np.zeros((3 * self.len_gradient_after_padding, 1))
        alpha = np.zeros((self.matrix_size, 1))
        
        for i in range(0, self.len_gradient_after_padding * 3 , 3 * self.matrix_size):
            tmp = np.dot(self.u_sigma, self.random_gradient_sum[i : i + 3 * self.matrix_size])
            for j in range(self.matrix_size): 
                alpha[j] = tmp[self.S_j[j]]
            res[int(i/3) : int(i/3) + self.matrix_size, :] = np.dot(self.A_inv, alpha)
        # set the gradient manually and update
        recovered_grad_in_list = trans2numpyArrayWithShapeList(res, self.shape_list)
        time2 = time.time()
        print('[dist]\t', np.sum(np.abs(res[:self.len_gradient, 0] - self.ori_gradient_sum)))
        print('ori ', self.ori_gradient_sum[:10])
        print("rec ", res[:10])
        print("time cost ", time2 - time1)
        return recovered_grad_in_list