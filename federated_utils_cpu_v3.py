import torch, threading
import time
import numpy as np
import math
import random
from scipy.linalg import null_space
import sys
#import pandas as pd
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
        res.append(torch.from_numpy(tmp.reshape(shape)).float().cuda())
        ind += listMulti(shape)
    return res

class Federated:
    def __init__(self, num_clients, matrix_size, num_threads, output_path):
        # utilize a sample gradient vector 
        self.num_clients = num_clients
        self.matrix_size = matrix_size
        self.num_threads = num_threads
        self.output_path = output_path
        self.MAX = 0.001
        self.S_i = random.sample(range(0, 3 * self.matrix_size), self.matrix_size)
        self.S_i.sort()
        self.S_j = random.sample(range(0, 2 * self.matrix_size), self.matrix_size)
        self.S_j.sort()
        self.all_index = random.sample(range(0, 3 * self.matrix_size), 3 * self.matrix_size)
        self.rand_index = list(set(self.all_index) - set(self.S_i))
        self.rand_index.sort()
        # dump indexing matrix
        np.save(self.output_path + 'S_i.npy', self.S_i)
        np.save(self.output_path + 'S_j.npy', self.S_j)
        #TODO
        self.all_gradient_var = []
        self.all_gradient_mean = []
        self.real_gradient_var = []
        self.real_gradient_mean = []
        self.rand_gradient_var = []
        self.rand_gradient_mean = []
        #self.ns_var = []
        #self.ns_var_var = []
        #self.ns_mean = []
        self.client_gradient_size = []
        self.client_real_gradient_size = []
        self.gradient_for_matrix_zero = 0
        self.client_time_elapsed = [[] for i in range(self.num_clients + 2)]
        self.init_time = 0
        self.server_time = 0
        # 1 initialization + 1 server + n client encrp + n client decrp 
    def init(self, gradient,shape_list):
        time1 = time.time()
        self.len_gradient = list(gradient.shape)[0]
        self.shape_list = shape_list
        self.len_gradient_after_padding = math.ceil(float(self.len_gradient) / (self.matrix_size * self.num_threads)) * self.matrix_size * self.num_threads
        #
        self.num_matrix = int(self.len_gradient_after_padding / self.matrix_size )
        self.A = []
        self.A_inv = []
        self.C = []
        self.vh = []
        self.u_sigma = []
        self.ns = []
        for i in range(self.num_matrix):
            self.A.append(self.MAX * np.random.rand(self.matrix_size, self.matrix_size))
            self.A_inv.append(np.linalg.inv(self.A[-1]))
            B = np.random.rand(self.matrix_size, 3 * self.matrix_size) * self.MAX
            for i in range(0, self.matrix_size):
                B[:, self.S_i[i] : self.S_i[i]+1] = self.A[-1][:, i:i+1]
            self.C.append(np.random.rand(2 * self.matrix_size, 3 * self.matrix_size) * self.MAX)
            for i in range(0, self.matrix_size):
                self.C[-1][self.S_j[i] : self.S_j[i] + 1, :] = B[i:i+1 , :]
        
            # SVD
            u, s, vh = np.linalg.svd(self.C[-1], full_matrices = True)
            #self.vh_t = np.transpose(self.vh) numpy doesn't need transpose for reconstruction!
            self.vh.append(vh)
            sigma = np.zeros((self.C[-1].shape[0], self.C[-1].shape[1]))
            sigma[: s.shape[0], : s.shape[0]] = np.diag(s)
            # null space
            self.u_sigma.append(np.dot(u, sigma))
            self.ns.append(null_space(self.C[-1])) # (3000, 1000) we use the first args.
        time2 = time.time()
        self.init_time = time2-time1
        self.trans_i = np.zeros((self.matrix_size, 3*self.matrix_size))
        self.trans_j = np.zeros((self.matrix_size, 2*self.matrix_size))
        for i,ind in  enumerate(self.S_i):
            self.trans_i[i][ind] = 1
        for i,ind in enumerate(self.S_j):
            self.trans_j[i][ind] = 1     
        self.trans_i_rand = np.zeros((2*self.matrix_size, 3*self.matrix_size))
        for i,ind in enumerate(self.rand_index):
            self.trans_i_rand[i][ind] = 1
        #self.ns_mean.append(np.mean(np.abs(self.ns)))
        #self.ns_var.append(np.mean(np.var(np.abs(self.ns), axis = 0)))
        #self.ns_var_var.append(np.var(np.var(np.abs(self.ns), axis = 0)))
    def init_gradient_sums(self):
        self.ori_gradient_sum = np.zeros((self.len_gradient))
        self.random_gradient_sum = np.zeros((self.len_gradient_after_padding * 3))
#        print("Initialization complete")
        
    def work_for_client(self, client_no, gradient):
        #print("Work for", client_no)
        time1 = time.time() #### start time
        part_num = self.len_gradient_after_padding / (self.num_threads * self.matrix_size)
        assert(client_no < self.num_clients)
        flatterned_grad = gradient.cpu().numpy()
        #flatterned_grad = transListOfArraysToArraysCpu(gradient, self.len_gradient)
        #call variance of real gradient
#        self.real_gradient_mean.append(np.mean(flatterned_grad))
#        self.real_gradient_var.append(np.var(flatterned_grad))
        
        self.ori_gradient_sum += flatterned_grad
        self.client_real_gradient_size.append(self.len_gradient)
        # padding
        flatterned_grad_extended = np.zeros((self.len_gradient_after_padding, 1)) # For zero padding
        flatterned_grad_extended[:self.len_gradient, 0] = flatterned_grad
        
        
        flatterned_grad_extended_after_random = self.MAX * np.random.rand(3 * self.len_gradient_after_padding, 1)
        ## TODO construct a transformation matrix, replace the assignment with matrix production
        def randomizing_matrix(thread_id, part_num):
            for i in range(int(thread_id * part_num * self.matrix_size), int((thread_id + 1) * part_num * self.matrix_size), self.matrix_size):
                #if (thread_id == 0 and int(i/self.matrix_size) % 50 == 0):
                #    print(thread_id, i/self.matrix_size, (int((thread_id + 1) * part_num)/self.matrix_size))
                #for j in range(0, self.matrix_size):
                #    flatterned_grad_extended_after_random[i * 3 + self.S_i[j]] = flatterned_grad_extended[i + j]
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
        # for i in range(0, self.len_gradient_after_padding, self.matrix_size):
        #     for j in range(0, self.matrix_size):
        #         flatterned_grad_extended_after_random[i * 3 + self.S_i[j]] = flatterned_grad_extended[i + j]
        
        # compute result
        #print("client ", client_no, " randomization complete")
        flatterned_grad_extended_final = self.MAX * np.random.rand(3 * self.len_gradient_after_padding, 1)
        ##TODO: optimize matrix calculation
        '''
        for i in range(0, self.len_gradient_after_padding, self.matrix_size):
            for j in range(0, self.matrix_size):
                flatterned_grad_extended_final[3 * i : 3*(i + self.matrix_size), :] = np.dot(np.transpose(self.vh), flatterned_grad_extended_after_random[3 * i : 3*(i + self.matrix_size), :] + kernel_space)
        '''
        def matrixProd(thread_id, part_num):
            for i in range(int(thread_id * part_num * self.matrix_size), int((thread_id + 1) * part_num * self.matrix_size), self.matrix_size):
                kernel_space = np.zeros((3 * self.matrix_size, 1))
                random_numbers = self.MAX * np.random.rand(self.matrix_size, 1)
                #TODO indepdent kernel space
                for j in range(self.matrix_size):
                    kernel_space += random_numbers[j] * self.ns[int(i/self.matrix_size)][:, j:j+1]
                flatterned_grad_extended_final[3 * i : 3*(i + self.matrix_size), :] \
                    = np.dot(self.vh[int(i / self.matrix_size)], flatterned_grad_extended_after_random[3 * i : 3*(i + self.matrix_size), :] + kernel_space)
                if (i == 0):
                    self.gradient_for_matrix_zero = flatterned_grad_extended_final[3 * i : 3*(i + self.matrix_size), :]
            #print(thread_id, " finish")
        threads = []
        for _i in range(self.num_threads):
            t = threading.Thread(target = matrixProd, args = (_i, part_num))
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()
        time2 = time.time() ### time end
        self.client_time_elapsed[client_no].append(time2-time1)
        time3 = time.time()
        self.random_gradient_sum += flatterned_grad_extended_final[:, 0]
        time4 = time.time()
        self.server_time += time4 - time3
        self.client_gradient_size.append(3*self.len_gradient_after_padding)
        self.all_gradient_mean.append(np.mean(flatterned_grad_extended_final[:, 0]))
        self.all_gradient_var.append(np.var(flatterned_grad_extended_final[:, 0]))
        
        real_pos_arr = np.zeros((self.len_gradient_after_padding))
        rand_pos_arr = np.zeros((2 * self.len_gradient_after_padding))
        def obtain_real_rand_pos(thread_id, part_num):
            for i in range(int(thread_id * self.matrix_size * part_num), int((thread_id+1) * self.matrix_size * part_num), self.matrix_size):
                real_pos_arr[i : i + self.matrix_size] = np.dot(flatterned_grad_extended_final[3 * i : 3*(i+self.matrix_size), :].reshape(1, 3*self.matrix_size), self.trans_i.T).reshape(self.matrix_size,)
                rand_pos_arr[2* i : 2 * i + 2*self.matrix_size] = np.dot(flatterned_grad_extended_final[3 * i : 3*(i+self.matrix_size), :].reshape(1, 3*self.matrix_size), self.trans_i_rand.T).reshape(2 * self.matrix_size,)
#                print(np.sum(real_pos_arr[i : i + self.matrix_size]), np.sum(rand_pos_arr[i : i + 2*self.matrix_size]), np.sum(flatterned_grad_extended_final[3 * i : 3*(i+self.matrix_size), :]))
                #(1*3n) (3n*n) = 1*n
        threads = []
        for _i in range(self.num_threads):
            t = threading.Thread(target = obtain_real_rand_pos, args = (_i, part_num))
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()
        self.rand_gradient_mean.append(np.mean(rand_pos_arr))
        self.rand_gradient_var.append(np.var(rand_pos_arr))
        self.real_gradient_mean.append(np.mean(real_pos_arr))
        self.real_gradient_var.append(np.var(real_pos_arr))
#        print(np.sum(real_pos_arr), np.sum(rand_pos_arr), np.sum(flatterned_grad_extended_final))
#        print(2*self.len_gradient_after_padding*self.rand_gradient_mean[-1] + self.len_gradient_after_padding*self.real_gradient_mean[-1], 3*self.len_gradient_after_padding*self.all_gradient_mean[-1])
#        self.rand_gradient_mean.append((flatterned_grad_extended_final[:,0].shape[0] * self.all_gradient_mean[-1] - flatterned_grad.shape[0] * self.real_gradient_mean[-1])/ ( flatterned_grad_extended_final[:,0].shape[0] -  flatterned_grad.shape[0]))
#        self.rand_gradient_mean.append((np.sum(flatterned_grad_extended_final) - np.sum(flatterned_grad))/( flatterned_grad_extended_final[:,0].shape[0] -  flatterned_grad.shape[0])) 
        # calculating the average of randomized gradient
        #mean = 0
        #for i in range(0, 3 * self.len_gradient_after_padding, 3 * self.matrix_size):
        #    for index in self.rand_index:
        #        mean += flatterned_grad_extended_final[i + index][0]
        #print(mean)
        #mean = mean / ( 3 * self.len_gradient_after_padding - self.len_gradient)
        #print('random mean', mean, self.rand_gradient_mean[-1])
#        print(3 * self.len_gradient_after_padding - self.len_gradient ,flatterned_grad_extended_final[:,0].shape[0] -  flatterned_grad.shape[0])
#        rand_gradient_var = 0
#        for i in range(0, 3 * self.len_gradient_after_padding, 3 * self.matrix_size):
#            for index in self.rand_index:
#                rand_gradient_var += (flatterned_grad_extended_final[index][0] - self.rand_gradient_mean[-1])**2
#        rand_gradient_var = rand_gradient_var / (3 * self.len_gradient_after_padding - self.len_gradient)
#        self.rand_gradient_var.append(rand_gradient_var)
        #print(self.rand_gradient_var[-1], self.real_gradient_var[-1], self.all_gradient_var[-1])
        #print("client ",  client_no, " masking complete",)
        #print("time for randomization ", time2 - time1, "time for masking", time3 - time2)
        
        #obtain real gradient array
#        real_grad_pos_arr = np.zeros((self.
    def recoverGradient(self):
        time1 = time.time()
        res = np.zeros((self.len_gradient_after_padding, 1))
        alpha = np.zeros((self.matrix_size, 1))
        
        for i in range(0, self.len_gradient_after_padding * 3 , 3 * self.matrix_size):
            tmp = np.dot(self.u_sigma[int(i/(3*self.matrix_size))], self.random_gradient_sum[i : i + 3 * self.matrix_size]) # (2n,1)
            alpha = np.dot(self.trans_j, tmp)
            alpha = np.reshape(alpha, (self.matrix_size,1))
            #for j in range(self.matrix_size): 
            #    alpha[j] = tmp[self.S_j[j]]
            res[int(i/3) : int(i/3) + self.matrix_size] = np.dot(self.A_inv[int(i/(3*self.matrix_size))], alpha)
        # set the gradient manually and update
        recovered_grad_in_list = trans2numpyArrayWithShapeList(res, self.shape_list)
        time2 = time.time()
        self.client_time_elapsed[self.num_clients].append(time2 - time1)
        self.client_time_elapsed[self.num_clients+1].append(self.server_time)
        self.server_time = 0
        #print('[dist]\t', np.sum(np.abs(res[:self.len_gradient, 0] - self.ori_gradient_sum)))
        #print('ori ', self.ori_gradient_sum[:10])
        #print("rec ", res[:10])
        #print("Recover gradient cost ", time2 - time1)
        return recovered_grad_in_list
    def writetxt(self, filename, l):
        f = open(filename, 'w')
        ll = [str(i)+'\n' for i in l]
        f.writelines(ll)
        f.close()
    def write_timetxt(self, filename, l):
        f = open(filename, 'w')
        assert(len(l) == self.num_clients + 2)
        print(len(l[0]), len(l[1]), len(l[2]), len(l[3]))
        for i in range(len(l[-1])):
            for j in range(self.num_clients + 2):
                f.write(str(l[j][i]) + '\t')
            f.write('\n')
    def dump(self):
        '''
        np.save(self.output_path + './all_gradient_mean.npy', np.array(self.all_gradient_mean))
        np.save(self.output_path + './all_gradient_var.npy', np.array(self.all_gradient_var))
        np.save(self.output_path +  np.array(self.rand_gradient_mean))
        np.save(self.output_path + './rand_gradient_var.npy', np.array(self.rand_gradient_var))
        np.save(self.output_path + './real_gradient_mean.npy', np.array(self.real_gradient_mean))
        np.save(self.output_path + './real_gradient_var.npy', np.array(self.real_gradient_var))
        np.save(self.output_path + './kernel_mean.npy', np.array(self.ns_mean))
        np.save(self.output_path + './kernel_var.npy', np.array(self.ns_var))
        np.save(self.output_path + './kernel_var_var.npy', np.array(self.ns_var_var))
        ''' 
        self.writetxt(self.output_path + './all_gradient_mean.txt', self.all_gradient_mean)
        self.writetxt(self.output_path + './all_gradient_var.txt', self.all_gradient_var)
        self.writetxt(self.output_path + './rand_gradient_mean.txt', self.rand_gradient_mean)
        self.writetxt(self.output_path + './rand_gradient_var.txt', self.rand_gradient_var)
        self.writetxt(self.output_path + './real_gradient_mean.txt',self.real_gradient_mean)
        self.writetxt(self.output_path + './real_gradient_var.txt', self.real_gradient_var)
        #self.writetxt(self.output_path + './kernel_mean.txt', self.ns_mean)
        #self.writetxt(self.output_path + './kernel_var.txt', self.ns_var)
        #self.writetxt(self.output_path + './kernel_var_var.txt', self.ns_var_var)  
        np.savetxt(self.output_path + './kernel_vector_0.txt', self.ns[0])
        np.savetxt(self.output_path + './gradient_for_matrix_0.txt', self.gradient_for_matrix_zero)
        self.writetxt(self.output_path + './client_grad_size.txt', self.client_gradient_size)
        self.writetxt(self.output_path + './client_real_grad_size.txt', self.client_real_gradient_size)
        self.write_timetxt(self.output_path + './time_elapsed.txt', self.client_time_elapsed)
        f = open(self.output_path + './time_init.txt', 'w')
        f.write(str(self.init_time) + '\n')
        f.close()
        print("successfully dumped")