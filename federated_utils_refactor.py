import torch, threading
import time
import numpy as np
import math
import random
from scipy.linalg import null_space


def listMulti(l):
        res = 1
        for ele in l:
            res *= ele
        return res
def transCPUarr2GPU(grad, shape_list):
    
    res = []
    ind = 0
    for shape in shape_list:
        tmp = grad[ind:ind+listMulti(shape)]
        res.append(torch.from_numpy(tmp.reshape(shape)).float().cuda())
        ind += listMulti(shape)
    return res
def transGPUarr2GPU(grad, shape_list):
    res = []
    ind = 0
    for shape in shape_list:
        tmp = grad[ind:ind+listMulti(shape)]
        tmp = tmp.view(shape)
        res.append(tmp.float())
        ind += listMulti(shape)
    return res
# CPU version
class Federated_CPU:

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
    def init(self, gradient,shape_list):
        self.len_gradient = list(gradient.shape)[0]
        self.shape_list = shape_list
        self.len_gradient_after_padding = math.ceil(float(self.len_gradient) / (self.matrix_size * self.num_threads)) * self.matrix_size * self.num_threads

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
        
        self.u, self.s, self.vh = np.linalg.svd(self.C, full_matrices = True)
        #no self.vh_t = np.transpose(self.vh) numpy doesn't need transpose for reconstruction!
        self.sigma = np.zeros((self.C.shape[0], self.C.shape[1]))
        self.sigma[: self.s.shape[0], : self.s.shape[0]] = np.diag(self.s)
        self.u_sigma = np.dot(self.u, self.sigma)
        self.ns = null_space(self.C) 
        self.trans_i = np.zeros((self.matrix_size, 3*self.matrix_size))
        self.trans_j = np.zeros((self.matrix_size, 2*self.matrix_size))
        for i,ind in  enumerate(self.S_i):
            self.trans_i[i][ind] = 1
        for i,ind in enumerate(self.S_j):
            self.trans_j[i][ind] = 1
    def work_for_client(self, client_no, gradient):
        part_num = self.len_gradient_after_padding / self.num_threads
        time1 = time.time()
        flatterned_grad = gradient.cpu().numpy()
        self.ori_gradient_sum += flatterned_grad
        # padding
        flatterned_grad_extended = np.zeros((self.len_gradient_after_padding, 1)) # For zero padding
        flatterned_grad_extended[:self.len_gradient, 0] = flatterned_grad
        kernel_space = np.zeros((3 * self.matrix_size, 1))
        random_numbers = self.MAX * np.random.rand(self.matrix_size, 1)

        for i in range(self.matrix_size):
            kernel_space += random_numbers[i] * self.ns[:, i:i+1]
        
        flatterned_grad_extended_after_random = self.MAX * np.random.rand(3 * self.len_gradient_after_padding, 1)
        def randomizing_matrix(thread_id, part_num):
            for i in range(int(thread_id * part_num), int((thread_id + 1) * part_num), self.matrix_size):
                flatterned_grad_extended_after_random[i * 3 : 3 * (i+self.matrix_size)] = \
                    (np.dot(flatterned_grad_extended[i : i + self.matrix_size].reshape(1, self.matrix_size), \
                        self.trans_i)).reshape(3*self.matrix_size,1)
        threads = []
        for _i in range(self.num_threads):
            t = threading.Thread(target = randomizing_matrix, args = (_i, part_num))
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()

        time2 = time.time()
        #print("client ", client_no, " randomization complete")
        flatterned_grad_extended_final = self.MAX * np.random.rand(3 * self.len_gradient_after_padding, 1)
        
        def matrixProd(thread_id, part_num):
            for i in range(int(thread_id * part_num), int((thread_id + 1) * part_num), self.matrix_size):
                flatterned_grad_extended_final[3 * i : 3*(i + self.matrix_size), :] \
                    = np.dot(self.vh, flatterned_grad_extended_after_random[3 * i : 3*(i + self.matrix_size), :] + kernel_space)
            #print(thread_id, " finish")
        threads = []
        part_num = self.len_gradient_after_padding / self.num_threads
        for _i in range(self.num_threads):
            t = threading.Thread(target = matrixProd, args = (_i, part_num))
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()
        self.random_gradient_sum += flatterned_grad_extended_final[:, 0]
        time3 = time.time()
        #print("client ",  client_no, " masking complete",)
        #print("time for randomization ", time2 - time1, "time for masking", time3 - time2)
    def recoverGradient(self):
        time1 = time.time()
        res = np.zeros((self.len_gradient_after_padding, 1))
        alpha = np.zeros((self.matrix_size, 1))
        
        for i in range(0, self.len_gradient_after_padding * 3 , 3 * self.matrix_size):
            tmp = np.dot(self.u_sigma, self.random_gradient_sum[i : i + 3 * self.matrix_size]) # (2n,1)
            alpha = np.dot(self.trans_j, tmp).reshape(alpha, (self.matrix_size,1))
            
            res[int(i/3) : int(i/3) + self.matrix_size] = np.dot(self.A_inv, alpha)
        # set the gradient manually and update
        recovered_grad_in_list = transCPUarr2GPU(res, self.shape_list)
        time2 = time.time()
        return recovered_grad_in_list

## GPU version
class Federated_GPU:
    def __init__(self, num_clients, matrix_size, num_threads):
        self.num_clients = num_clients
        self.matrix_size = matrix_size
        self.num_threads = num_threads
        self.MAX = 0.001
        self.S_i = random.sample(range(0, 3 * self.matrix_size), self.matrix_size)
        self.S_i.sort()
        self.S_j = random.sample(range(0, 2 * self.matrix_size), self.matrix_size)
        self.S_j.sort()
        
    def init(self, gradient, shape_list):
        #TODO each round iniitialization cost too much !
        self.len_gradient = list(gradient.shape)[0]
        self.shape_list = shape_list
        self.len_gradient_after_padding = math.ceil(float(self.len_gradient) / (self.matrix_size * self.num_threads)) * self.matrix_size * self.num_threads
        self.part_num = self.len_gradient_after_padding / self.num_threads

        self.ori_gradient_sum = torch.zeros(self.len_gradient).cuda()
        self.random_gradient_sum = torch.zeros(self.len_gradient_after_padding * 3).cuda()
        
        self.A = self.MAX * torch.rand(self.matrix_size, self.matrix_size).float().cuda()
        self.A_inv = self.A.inverse()
        self.B = torch.zeros(self.matrix_size, 3 * self.matrix_size).cuda()
        for i in range(0, self.matrix_size):
            self.B[:, self.S_i[i] : self.S_i[i]+1] = self.A[:, i:i+1]
        self.C = (torch.rand(2 * self.matrix_size, 3 * self.matrix_size) * self.MAX).float().cuda()
        for i in range(0, self.matrix_size):
            self.C[self.S_j[i] : self.S_j[i] + 1, :] = self.B[i:i+1 , :]
        
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
        time1 = time.time()
        flatterned_grad = gradient
        self.ori_gradient_sum += flatterned_grad
        flatterned_grad_extended = torch.zeros(self.len_gradient_after_padding, 1).cuda()
        flatterned_grad_extended[:self.len_gradient, 0] = flatterned_grad
        kernel_space = torch.zeros(3 * self.matrix_size, 1).cuda()
        random_numbers = self.MAX * torch.rand(self.matrix_size, 1).cuda()
        
        for i in range(self.matrix_size):
            kernel_space += random_numbers[i] * self.ns[:, i:i+1]
        #print("kernel space complete")
        flatterned_grad_extended_after_random = (self.MAX * torch.rand(3 * self.len_gradient_after_padding, 1)).float().cuda()

        def randomizing_matrix(thread_id, part_num):
            for i in range(int(thread_id * part_num), int((thread_id + 1) * part_num), self.matrix_size):
                flatterned_grad_extended_after_random[i * 3 : 3 * (i + self.matrix_size)] = \
                    (torch.mm(flatterned_grad_extended[i : i + self.matrix_size].view(1, self.matrix_size), \
                        self.trans_i)).view(3*self.matrix_size,1)
        threads = []
        for _i in range(self.num_threads):
            t = threading.Thread(target = randomizing_matrix, args = (_i, self.part_num))
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()
        # compute result
        time2 = time.time()
        flatterned_grad_extended_final = (self.MAX * torch.rand(3 * self.len_gradient_after_padding, 1)).float().cuda()
        ###TODO: combine two process together
        def matrixProd(thread_id, part_num):
            
            for i in range(int(thread_id * part_num), int((thread_id + 1) * part_num), self.matrix_size):
                flatterned_grad_extended_final[3 * i : 3*(i + self.matrix_size), :] \
                    = torch.mm(self.vh_t, flatterned_grad_extended_after_random[3 * i : 3*(i + self.matrix_size), :] + kernel_space)
            #print(thread_id, " finish")
        threads = []
        for _i in range(self.num_threads):
            t = threading.Thread(target = matrixProd, args = (_i, self.part_num))
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()
    
        self.random_gradient_sum += flatterned_grad_extended_final[:, 0]
        time3 = time.time()
    def recoverGradient(self):
        time1 = time.time()
        res = torch.zeros(self.len_gradient_after_padding, 1).cuda()
        alpha = torch.zeros(self.matrix_size, 1).cuda()
        
        for i in range(0, self.len_gradient_after_padding * 3 , 3 * self.matrix_size):
            tmp = torch.mm(self.u_sigma, self.random_gradient_sum[i : i + 3 * self.matrix_size].view(-1,1))
            alpha = torch.mm(self.trans_j, tmp)
            res[int(i/3) : int(i/3) + self.matrix_size, :] = torch.mm(self.A_inv, alpha)
        # set the gradient manually and update
        recovered_grad_in_cuda = transGPUarr2GPU(res, self.shape_list)
        time2 = time.time()
        return recovered_grad_in_cuda