# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import time
from time import perf_counter as pc
import threading

class ProbeCudaMem(object):
	def __init__(self, rank, period=0.01): # sec
		self.rank = int(rank)
		self.period = float(period)
		assert torch.cuda.is_available()

	def probe(self):
		# # # read memory stats
		mem_stats = torch.cuda.memory_stats(self.rank) # cpu op
		# mem_alloc = mem_stats["allocated_bytes.all.current"] # torch.cuda.memory_allocated
		mem_alloc_peak = mem_stats["allocated_bytes.all.peak"] # torch.cuda.max_memory_allocated
		# mem_resrv = mem_stats["reserved_bytes.all.current"] # torch.cuda.memory_reserved
		mem_resrv_peak = mem_stats["reserved_bytes.all.peak"] # torch.cuda.max_memory_reserved
		# # # timestamp memory stats
		torch.cuda.nvtx.mark("%d/%d"%(mem_alloc_peak,mem_resrv_peak)) # cpu op
		# # # reset for next probe
		torch.cuda.reset_peak_memory_stats(self.rank) # cpu op, has lock

	def _loop_probe(self):
		starttime = pc() # sec	
		while True:
			if self.stop_probe:
				break
			self.probe()
			time.sleep(self.period - ((pc()-starttime) % self.period)) # sec
		# ref: https://stackoverflow.com/questions/474528/what-is-the-best-way-to-repeatedly-execute-a-function-every-x-seconds
		# > it works with perfect intervels of 10ms
		# > but can skip some intervals when CPU is too busy (due to python GIL)

	def start(self):
		print("rank%d: ProbeCudaMem starts (period = %.1f ms)"%(self.rank, self.period*1000.))
		self.stop_probe = False
		self.thread = threading.Thread(target=self._loop_probe)
		self.thread.start()

	def stop(self):
		self.stop_probe = True
		self.thread.join()
		print("rank%d: ProbeCudaMem stops (period = %.1f ms)"%(self.rank, self.period*1000.))

# # # # example code below # # #
# import os
# import torch.distributed as dist # multi-process over multi-node
# from torch.multiprocessing import Process # multi-process on single node with shared memory
# import numpy as np
# import torch.cuda.profiler as cuda_profiler #===
# from torch.cuda.nvtx import mark as nvtx_mark #===
# from torch.cuda.nvtx import range_push as nvtx_range_push #===
# from torch.cuda.nvtx import range_pop as nvtx_range_pop #===

# def test_run(rank, size):
# 	torch.cuda.set_device(rank)
# 	pgroup = dist.new_group([0, 1]) # NCCL broadcast's group
# 	print("rank%d (pid%d): new_group'ed"% (rank, os.getpid()))
	
# 	tensor_size = 10**9 # [1, 10**3, 10**6, 10**9]: # Byte
# 	print("tensor_size={} Byte".format(tensor_size))
	
# 	torch.cuda.synchronize(rank)
# 	dist.barrier()
	
# 	# global thread_stop; thread_stop = False
# 	# thread = threading.Thread(target=probe_thread, args=(rank,))
# 	# thread.start()
# 	probe_cuda_mem = ProbeCudaMem(rank)
# 	probe_cuda_mem.start() #~~ 

# 	print("rank%d: cuda profiler starts"%rank)
# 	cuda_profiler.start()
# 	nvtx_mark("cudaProfilerStart") #===

# 	probe_cuda_mem.probe() #~~
# 	tensor_cpu = torch.zeros(tensor_size, dtype=torch.int8, pin_memory=True)
# 	nvtx_range_push("SwapIn") #===
# 	tensor_gpu = tensor_cpu.cuda(rank)
# 	nvtx_range_pop() #===
# 	probe_cuda_mem.probe() #~~
# 	# probe_cuda_memory() #~~ 

# 	nvtx_range_push("Addition") #===
# 	tensor_gpu += 1
# 	nvtx_range_pop() #===
	
# 	nvtx_range_push("Multiplication") #===
# 	tensor_gpu *= 10
# 	nvtx_range_pop() #===
	
# 	nvtx_range_push("Multiplication") #===
# 	tensor_gpu *= 100
# 	nvtx_range_pop() #===
	
# 	nvtx_range_push("Multiplication") #===
# 	tensor_gpu *= 1000
# 	nvtx_range_pop() #===
# 	nvtx_mark("ComputeEnds")
	
# 	nvtx_range_push("Bcast") #===
# 	dist.broadcast(tensor=tensor_gpu, group=pgroup, src=0)
# 	nvtx_range_pop() #===
# 	nvtx_mark("P2PEnds")
	
# 	probe_cuda_mem.probe() #~~
# 	nvtx_range_push("SwapOut") #===
# 	tensor_cpu = tensor_gpu.cpu()
# 	nvtx_range_pop() #===
# 	probe_cuda_mem.probe() #~~
# 	# probe_cuda_memory() #~~ 
	
# 	torch.cuda.synchronize(rank)
# 	dist.barrier()

# 	nvtx_mark("cudaProfilerStop") #===
# 	cuda_profiler.stop()
# 	print("rank%d: cuda profiler stops"%rank)
	
# 	# thread_stop = True
# 	# thread.join()
# 	probe_cuda_mem.stop() #~~

# 	dist.barrier()
# 	print("Done")
	
# def test_init_process(rank, size, fn, backend='nccl'):
#     """ Initialize the distributed environment. """
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29500'
#     dist.init_process_group(backend, rank=rank, world_size=size)
#     fn(rank, size)

# if __name__ == "__main__": # TEST
#     size = 2
#     processes = []
#     for rank in range(size):
#         p = Process(target=test_init_process, args=(rank, size, test_run))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()
