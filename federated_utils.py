import torch, time
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
    def __init__(self, num_clients, matrix_size):
        # utilize a sample gradient vector 
        self.num_clients = num_clients
        self.matrix_size = matrix_size
        self.MAX = 0.001
        self.S_i = random.sample(range(0, 3 * self.matrix_size), self.matrix_size)
        self.S_i.sort()
        self.S_j = random.sample(range(0, 2 * self.matrix_size), self.matrix_size)
        self.S_j.sort()
        self.num_threads = 10
    def init(self, gradient):
        self.len_gradient, self.shape_list = get_shape_and_length_gradient_cuda(gradient)
        self.len_gradient_after_padding = 3 * math.ceil(float(self.len_gradient)/self.matrix_size) * self.matrix_size
        #self.len_gradient_after_padding = math.ceil(float(self.len_gradient)/self.matrix_size) * self.matrix_size
        print(self.len_gradient_after_padding)
        self.ori_gradient_sum = torch.zeros(self.len_gradient).cuda()
        self.random_gradient_sum = torch.zeros(self.len_gradient_after_padding * 3).cuda()
        #self.random_gradient_sum = torch.zeros(self.len_gradient_after_padding).cuda()
        
        #self.A = torch.randint(0, 0.001, (self.matrix_size, self.matrix_size)).float().cuda()
        self.A = self.MAX * torch.rand(self.matrix_size, self.matrix_size).float().cuda()
        self.A_inv = self.A.inverse()
        self.B = torch.zeros(self.matrix_size, 3 * self.matrix_size).cuda()
        for i in range(0, self.matrix_size):
            self.B[:, self.S_i[i] : self.S_i[i]+1] = self.A[:, i:i+1]
        self.C = (torch.rand(2 * self.matrix_size, 3 * self.matrix_size) * self.MAX).float().cuda()
        for i in range(0, self.matrix_size):
            self.C[self.S_j[i] : self.S_j[i] + 1, :] = self.B[i:i+1 , :]
        
        # test 
        #self.A = self.C = torch.randint(0, self.MAX, (self.matrix_size, self.matrix_size)).float().cuda()
        #self.A_inv = self.A.inverse()
        # SVD
        self.u, self.s, self.vh = torch.svd(self.C, some=False)
        self.vh_t = self.vh.t()
        self.sigma = torch.zeros(self.C.shape[0], self.C.shape[1]).cuda()
        self.sigma[: self.s.shape[0], : self.s.shape[0]] = self.s.diag()
        # null space
        self.u_sigma = torch.mm(self.u, self.sigma)
        ns = null_space(self.C.cpu()) # (3000, 1000) we use the first args.
        self.ns = torch.from_numpy(ns).cuda()
        print("Initializatin complete", self.ns.shape)
    def work_for_client(self, client_no, gradient):
        assert(client_no < self.num_clients)
        time1 = time.time()
        flatterned_grad = transListOfArraysToArraysCuda(gradient, self.len_gradient)
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
        
        flatterned_grad_extended_after_random = (self.MAX * torch.rand(3 * self.len_gradient_after_padding, 1)).float().cuda()

        for i in range(0, self.len_gradient_after_padding, self.matrix_size):
            for j in range(0, self.matrix_size):
                flatterned_grad_extended_after_random[i * 3 + self.S_i[j]] = flatterned_grad_extended[i + j]
        
        # compute result
        time2 = time.time()
        print("client ", client_no, " randomization complete")
        flatterned_grad_extended_final = (self.MAX * torch.rand(3 * self.len_gradient_after_padding, 1)).float().cuda()
        ###TODO: multi threading
        for i in range(0, self.len_gradient_after_padding, self.matrix_size):
            for j in range(0, self.matrix_size): 
                flatterned_grad_extended_final[3 * i : 3*(i + self.matrix_size), :] = torch.mm(self.vh_t, flatterned_grad_extended_after_random[3 * i : 3*(i + self.matrix_size), :] + kernel_space)

        self.random_gradient_sum += flatterned_grad_extended_final[:, 0]
        time3 = time.time()
        #self.random_gradient_sum += torch.randn(3 * self.len_gradient_after_padding).cuda() * 0.0001
        print("client ",  client_no, " masking complete",)
        print("time for randomization ", time2 - time1, "time for masking", time3 - time2)
    def recoverGradient(self):
        time1 = time.time()
        res = torch.zeros(3 * self.len_gradient_after_padding, 1).cuda()
        alpha = torch.zeros(self.matrix_size, 1).cuda()
        
        for i in range(0, self.len_gradient_after_padding * 3 , 3 * self.matrix_size):
            ###TODO: time issue
            tmp = torch.mm(self.u_sigma, self.random_gradient_sum[i : i + 3 * self.matrix_size].view(-1,1))
            for j in range(self.matrix_size): 
                alpha[j] = tmp[self.S_j[j]]
            res[int(i/3) : int(i/3) + self.matrix_size, :] = torch.mm(self.A_inv, alpha)
        ''' 
        for i in range(0, self.len_gradient_after_padding , self.matrix_size):
            tmp = torch.mm(torch.mm(self.u, self.sigma), self.random_gradient_sum[i : i + self.matrix_size].view(-1,1))
            for j in range(self.matrix_size): 
                #alpha[j] = tmp[self.S_j[j]]
                alpha[j] = tmp[j]
            #res[int(i/3) : int(i/3) + self.matrix_size, :] = torch.mm(self.A_inv, alpha)
            res[i : i + self.matrix_size, :] = torch.mm(self.A_inv, alpha)
        '''
        # set the gradient manually and update
        recovered_grad_in_cuda = transCudaArrayWithShapeList(res, self.shape_list)
        #print('[dist]\t', torch.dist(res[:self.len_gradient], self.ori_gradient_sum))
        time2 = time.time()
        print('[dist]\t', torch.sum(torch.abs(res[:self.len_gradient, 0] - self.ori_gradient_sum)))
        #self.ori_gradient_sum.fill_(0)
        #self.random_gradient_sum.fill_(0)
        print('ori ', self.ori_gradient_sum.view(-1, 1)[:10])
        print("rec ", recovered_grad_in_cuda[0].view(-1,1)[:10])
        print("recover time cost ", time2 - time1)
        return recovered_grad_in_cuda