import torch, time, threading
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
def get_shape_and_length_gradient_cuda(current_grad):
    n = 0
    res = []
    for arr in current_grad:
        n += list(arr.view(-1,).shape)[0]
        res.append(arr.shape)
    return n, res
def transListOfArraysToArraysCuda(current_grad, n):
    # expect a list of arrays, return a squeezed cuda array, n is the total length
    res = torch.zeros(n).cuda()
    ind = 0
    for arr in current_grad:
        arr = arr.view(-1,)
        res[ind:ind+arr.shape[0]] = arr 
        ind+=arr.shape[0]
    return res
def listMultiCuda(l):
    res = 1
    for ele in l:
        res *= ele
    return res
def transCudaArrayWithShapeList(grad, shape_list):
    res = []
    ind = 0
    for shape in shape_list:
        tmp = grad[ind:ind+listMulti(shape)]
        tmp = tmp.view(shape)
        res.append(tmp.float())
        ind += listMulti(shape)
    return res

class Federated:
    def __init__(self, num_clients, matrix_size, num_threads):
        # utilize a sample gradient vector 
        self.num_clients = num_clients
        self.matrix_size = matrix_size
        self.num_threads = num_threads
        self.MAX = 0.001
        self.S_i = random.sample(range(0, 3 * self.matrix_size), self.matrix_size)
        self.S_i.sort()
        self.S_j = random.sample(range(0, 2 * self.matrix_size), self.matrix_size)
        self.S_j.sort()
    def init(self, gradient, shape_list):
        self.len_gradient = list(gradient.shape)[0]
        self.shape_list = shape_list
        #print("gradient length", self.len_gradient)
        #self.len_gradient, self.shape_list = get_shape_and_length_gradient_cuda(gradient)
        self.len_gradient_after_padding = math.ceil(float(self.len_gradient) / (self.matrix_size * self.num_threads)) * self.matrix_size * self.num_threads
        
        self.ori_gradient_sum = torch.zeros(self.len_gradient).cuda()
        #print(self.ori_gradient_sum.shape)
        self.random_gradient_sum = torch.zeros(self.len_gradient_after_padding * 3).cuda()
        
        self.A = self.MAX * torch.rand(self.matrix_size, self.matrix_size).float().cuda()
        self.A_inv = self.A.inverse()
        self.B = torch.zeros(self.matrix_size, 3 * self.matrix_size).cuda()
        for i in range(0, self.matrix_size):
            self.B[:, self.S_i[i] : self.S_i[i]+1] = self.A[:, i:i+1]
        self.C = (torch.rand(2 * self.matrix_size, 3 * self.matrix_size) * self.MAX).float().cuda()
        for i in range(0, self.matrix_size):
            self.C[self.S_j[i] : self.S_j[i] + 1, :] = self.B[i:i+1 , :]
        
        # SVD
        self.u, self.s, self.vh = torch.svd(self.C, some=False)
        self.vh_t = self.vh.t()
        self.sigma = torch.zeros(self.C.shape[0], self.C.shape[1]).cuda()
        self.sigma[: self.s.shape[0], : self.s.shape[0]] = self.s.diag()
        # null space
        self.u_sigma = torch.mm(self.u, self.sigma)
        ns = null_space(self.C.cpu()) # (3000, 1000) we use the first args.
        self.ns = torch.from_numpy(ns).cuda()
        self.trans_i = torch.zeros(self.matrix_size, 3*self.matrix_size).cuda()
        self.trans_j = torch.zeros(self.matrix_size, 2*self.matrix_size).cuda()
        for i,ind in  enumerate(self.S_i):
            self.trans_i[i][ind] = 1
        for i,ind in enumerate(self.S_j):
            self.trans_j[i][ind] = 1
        #print("initialization complete")
    def work_for_client(self, client_no, gradient):
        assert(client_no < self.num_clients)
        part_num = self.len_gradient_after_padding / self.num_threads
        time1 = time.time()
        ### TODO time issue
        flatterned_grad = gradient
        self.ori_gradient_sum += flatterned_grad
        # padding
        flatterned_grad_extended = torch.zeros(self.len_gradient_after_padding, 1).cuda()
        flatterned_grad_extended[:self.len_gradient, 0] = flatterned_grad
        kernel_space = torch.zeros(3 * self.matrix_size, 1).cuda()
        random_numbers = self.MAX * torch.rand(self.matrix_size, 1).cuda()
        ###TODO
        #kernel_space = torch.zeros(self.matrix_size, 1).cuda()
        for i in range(self.matrix_size):
            kernel_space += random_numbers[i] * self.ns[:, i:i+1]
        #print("kernel space complete")
        flatterned_grad_extended_after_random = (self.MAX * torch.rand(3 * self.len_gradient_after_padding, 1)).float().cuda()

        def randomizing_matrix(thread_id, part_num):
            for i in range(int(thread_id * part_num), int((thread_id + 1) * part_num), self.matrix_size):
                #if (thread_id == 0):
                #    print(thread_id, i/self.matrix_size, (int((thread_id + 1) * part_num)/self.matrix_size))
                flatterned_grad_extended_after_random[i * 3 : 3 * (i + self.matrix_size)] = \
                    (torch.mm(flatterned_grad_extended[i : i + self.matrix_size].view(1, self.matrix_size), \
                        self.trans_i)).view(3*self.matrix_size,1)
        threads = []
        for _i in range(self.num_threads):
            t = threading.Thread(target = randomizing_matrix, args = (_i, part_num))
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()
        # compute result
        time2 = time.time()
        #print("client ", client_no, " randomization complete ", time2 - time1)
        flatterned_grad_extended_final = (self.MAX * torch.rand(3 * self.len_gradient_after_padding, 1)).float().cuda()
        ###TODO: multi threading
        def matrixProd(thread_id, part_num):
            
            for i in range(int(thread_id * part_num), int((thread_id + 1) * part_num), self.matrix_size):
                flatterned_grad_extended_final[3 * i : 3*(i + self.matrix_size), :] \
                    = torch.mm(self.vh_t, flatterned_grad_extended_after_random[3 * i : 3*(i + self.matrix_size), :] + kernel_space)
            #print(thread_id, " finish")
        threads = []
        for _i in range(self.num_threads):
            t = threading.Thread(target = matrixProd, args = (_i, part_num))
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()

        #for i in range(0, self.len_gradient_after_padding, self.matrix_size):
        #    flatterned_grad_extended_final[3 * i : 3*(i + self.matrix_size), :] = torch.mm(self.vh_t, flatterned_grad_extended_after_random[3 * i : 3*(i + self.matrix_size), :] + kernel_space)

        self.random_gradient_sum += flatterned_grad_extended_final[:, 0]
        time3 = time.time()
        #self.random_gradient_sum += torch.randn(3 * self.len_gradient_after_padding).cuda() * 0.0001
        #print("client ",  client_no, " masking complete",)
        #print("time for masking", time3 - time2)
    def recoverGradient(self):
        time1 = time.time()
        res = torch.zeros(self.len_gradient_after_padding, 1).cuda()
        alpha = torch.zeros(self.matrix_size, 1).cuda()
        
        for i in range(0, self.len_gradient_after_padding * 3 , 3 * self.matrix_size):
            ###TODO: time issue
            tmp = torch.mm(self.u_sigma, self.random_gradient_sum[i : i + 3 * self.matrix_size].view(-1,1))
            alpha = torch.mm(self.trans_j, tmp)
            res[int(i/3) : int(i/3) + self.matrix_size, :] = torch.mm(self.A_inv, alpha)
        # set the gradient manually and update
        recovered_grad_in_cuda = transCudaArrayWithShapeList(res, self.shape_list)
        #print('[dist]\t', torch.dist(res[:self.len_gradient], self.ori_gradient_sum))
        time2 = time.time()
        #print('[dist]\t', torch.sum(torch.abs(res[:self.len_gradient, 0] - self.ori_gradient_sum)))
        #self.ori_gradient_sum.fill_(0)
        #self.random_gradient_sum.fill_(0)
       # print('ori ', self.ori_gradient_sum.view(-1, 1)[:10])
        #print("rec ", recovered_grad_in_cuda[0].view(-1,1)[:10])
        #print("recover time cost ", time2 - time1)
        return recovered_grad_in_cuda
