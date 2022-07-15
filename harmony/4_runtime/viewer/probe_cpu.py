# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import time
from time import perf_counter as pc
import threading
import psutil

"""
System-wide CPU check (no affected by cpu binding):
1. logical processor utilization%: psutil.cpu_percent(interval=1, percpu=True)
2. # logical processor in total: psutil.cpu_count()

Per-process CPU check (affected by cpu binding) (numactl or docker run): 
p = psutil.Process()
1. logical processor utilization%: p.cpu_percent() 
2. logical processor affinity: p.cpu_affinity()
3. # logical processor running: p.cpu_num()

Experiences:
-. system-wide cpu_percent(interval=10ms is great) but < 3ms becomes zero
-. per-process cpu_percent(interval=10ms is decent) but can exceed number of cpus
	--> set minimal effective interval && clip exceedings && increase interval
"""

class ProbeCPU(object):
	def __init__(self, pids, ranks, period=0.01, min_interval=0.004): # sec
		assert len(pids) == len(ranks), "pid list and rank list must match"
		self.pids = pids
		self.ranks = ranks
		self.procs = [psutil.Process(int(pid)) for pid in pids]
		self.period = float(period)
		self.min_interval = float(min_interval)
		# check self affinity
		self.cpu_affinity = sorted(psutil.Process().cpu_affinity())
		self.cpu_affinity_cnt = len(self.cpu_affinity)
		print("[ProbeCPU] System has {} logical processors. Program uses {} ({}) (i.e., {}).".format( 
			psutil.cpu_count(), 
			self.cpu_affinity_cnt, 
			"%.1f%%"%(self.cpu_affinity_cnt/float(psutil.cpu_count())*100.), 
			self.cpu_affinity)) 
		# check affinity inheritage
		for pid, rank, p in zip(self.pids, self.ranks, self.procs):
			p_affinity = sorted(p.cpu_affinity())
			p_num = p.cpu_num()
			print("[ProbeCPU] rank{}'s logical processor affinity = {}, logical processor number = {}".format(rank, p_affinity, p_num))
			# assert self.cpu_affinity == p_affinity, "cpu affinity must inherit"
	
	def _start(self):
		print("[ProbeCPU] starts (period %.1f ms) (min_interval %.1f ms)"%(self.period*1000., self.min_interval*1000.))
		torch.cuda.profiler.start()
		torch.cuda.nvtx.mark("cudaProfilerStart") #===
		# torch.cuda.nvtx.range_push("ProbingCPU") # DEBUG

	def _probe(self):
		# # # read cpu stats
		per_p_percent = [ "%.1f" % p.cpu_percent(interval=None) for p in self.procs ] # non-blocking (percentage% since last call)
		sys_cpu_percent = psutil.cpu_percent(interval=None, percpu=True) # First element of the list refers to first logical CPU, second element to second logical CPU and so on. 
		total_cpu_percent = sum([sys_cpu_percent[i] for i in self.cpu_affinity]) # list comprehension 2.5x faster than the loop
		# # # timestamp cpu stats: "total_percent|p0_percent,p1_percent|total_cnt"
		torch.cuda.nvtx.mark("%.1f|%s|%d"%(
							total_cpu_percent, 
							",".join(per_p_percent), 
							self.cpu_affinity_cnt)) # cpu op
	
	def _stop(self):
		# torch.cuda.nvtx.range_pop() # DEBUG
		torch.cuda.nvtx.mark("cudaProfilerStop") #===
		torch.cuda.profiler.stop()
		print("[ProbeCPU] stops (period %.1f ms) (min_interval %.1f ms)"%(self.period*1000., self.min_interval*1000.))

	def run(self, process): 
		""" Probe all the time """ 
		starttime = pc() # sec	
		self._start()
		while process.is_alive():
			self._probe()
			interval = self.period - ((pc()-starttime) % self.period)
			if interval < self.min_interval:
				interval += self.period
			time.sleep(interval) # sec
		self._stop()
		# ref: https://stackoverflow.com/questions/474528/what-is-the-best-way-to-repeatedly-execute-a-function-every-x-seconds
		# > only probe all the time works (i.e., using shared variable or queue for start and end timestamp doesn't work)

	# def run(self, shared_flag): # works but is equal to probe all the time
	# 	assert shared_flag.value == 0
	# 	prev_shared_flag = 0
	# 	starttime = pc() # sec	
	# 	self._start()
	# 	while True:
	# 		if shared_flag.value == 2: # end
	# 			break
	# 		self._probe()
	# 		time.sleep(self.period - ((pc()-starttime) % self.period)) # sec
	# 	self._stop()
	# 	# ref: https://stackoverflow.com/questions/474528/what-is-the-best-way-to-repeatedly-execute-a-function-every-x-seconds

	# def run(self, shared_flag): # 10x slow down & start after child finishes
	# 	assert shared_flag.value == 0
	# 	prev_status = 0
	# 	starttime = pc() # sec	
	# 	while True:
	# 		status = shared_flag.value
	# 		if status == 0: # idle
	# 			prev_status = 0 
	# 		elif status == 1: # probe
	# 			if prev_status == 0:
	# 				self._start()
	# 			self._probe()
	# 			prev_status = 1
	# 		elif status == 2: # end
	# 			self._stop()
	# 			# if prev_status == 1:
	# 			# 	self._stop()
	# 			# prev_status = 2
	# 			break
	# 		else:
	# 			assert False
	# 		time.sleep(self.period - ((pc()-starttime) % self.period)) # sec


	# def run(self, shared_queue): # same 10x slow down & start after child finish
	# 	assert shared_queue.empty()
	# 	status = shared_queue.get() # blocking
	# 	assert status == 1
	# 	self._start()
	# 	starttime = pc() # sec	
	# 	while True:
	# 		if not shared_queue.empty(): # end
	# 			break
	# 		self._probe()
	# 		time.sleep(self.period - ((pc()-starttime) % self.period)) # sec
	# 	self._stop()
	# 	assert shared_queue.get() == 2
	
	
# # # # example code below # # #
# import os
# import torch.distributed as dist # multi-process over multi-node
# mp = torch.multiprocessing.get_context('spawn') # for GPU usage
# print("mp.get_start_method={}".format(mp.get_start_method()))
# import numpy as np
# import torch.cuda.profiler as cuda_profiler #===
# from torch.cuda.nvtx import mark as nvtx_mark #===
# from torch.cuda.nvtx import range_push as nvtx_range_push #===
# from torch.cuda.nvtx import range_pop as nvtx_range_pop #===
# import time
# from probe_cuda_mem import ProbeCudaMem

# def test_run(rank, size):
# 	torch.cuda.set_device(rank)
# 	pgroup = dist.new_group([0, 1]) # NCCL broadcast's group
# 	print("rank%d (pid%d): new_group'ed"% (rank, os.getpid()))
	
# 	tensor_size = 300*10**6 # [1, 10**3, 10**6, 10**9]: # Byte
# 	print("tensor_size={} Byte".format(tensor_size))
	
# 	torch.cuda.synchronize(rank)
# 	dist.barrier()
# 	t_start = time.time()

# 	# if rank == 0: 
# 	# 	shared_probecpu_queue.put(1) # !!!
# 	# 	print("rank%d: shared_probecpu_queue.put(1)"%rank)
# 	# 	# shared_probecpu_flag.value = 1 # !!!
# 	# 	# print("rank%d: shared_probecpu_flag=1"%rank)

# 	probe_cuda_mem = ProbeCudaMem(rank)
# 	probe_cuda_mem.start() #~~ 
	
# 	print("rank%d: cuda profiler starts"%rank)
# 	cuda_profiler.start()
# 	nvtx_mark("cudaProfilerStart") #===

# 	tensor_cpu = torch.zeros(tensor_size, dtype=torch.int8, pin_memory=True)

# 	nvtx_range_push("CAddition") #===
# 	tensor_cpu += 1
# 	nvtx_range_pop() #===
	
# 	nvtx_range_push("CMultiplication") #===
# 	tensor_cpu *= 10
# 	nvtx_range_pop() #===
	
# 	nvtx_range_push("CMultiplication") #===
# 	tensor_cpu *= 100
# 	nvtx_range_pop() #===
	
# 	nvtx_range_push("CMultiplication") #===
# 	tensor_cpu *= 1000
# 	nvtx_range_pop() #===
	
# 	nvtx_range_push("SwapIn") #===
# 	tensor_gpu = tensor_cpu.cuda(rank)
# 	nvtx_range_pop() #===
	
# 	nvtx_range_push("Addition") #===
# 	tensor_gpu += 1
# 	nvtx_range_pop() #===

# 	nvtx_range_push("Bcast") #===
# 	dist.broadcast(tensor=tensor_gpu, group=pgroup, src=0)
# 	nvtx_range_pop() #===
	
# 	nvtx_range_push("SwapOut") #===
# 	tensor_cpu = tensor_gpu.cpu()
# 	nvtx_range_pop() #===
	
# 	torch.cuda.synchronize(rank)
# 	dist.barrier()

# 	nvtx_mark("cudaProfilerStop") #===
# 	cuda_profiler.stop()
# 	print("rank%d: cuda profiler stops"%rank)

# 	probe_cuda_mem.stop() #~~

# 	# if rank == 0:
# 	# 	shared_probecpu_queue.put(2) # !!!
# 	# 	print("rank%d: shared_probecpu_queue.put(2)"%rank) 
# 	# 	# shared_probecpu_flag.value = 2 # !!!
# 	# 	# print("rank%d: shared_probecpu_flag=2"%rank)

# 	t_end = time.time()
# 	dist.barrier()
# 	print("Done %.6f sec"%(t_end-t_start))
	
# def test_init_process(rank, size, fn, backend='nccl'):
#     """ Initialize the distributed environment. """
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29500'
#     dist.init_process_group(backend, rank=rank, world_size=size)
#     fn(rank, size)

# if __name__ == "__main__": # TEST
# 	size = 2
# 	processes = []
# 	# shared_probecpu_flag = mp.Value('i', 0) # 0: idle, 1: probing, 2: end
# 	# shared_probecpu_queue = mp.SimpleQueue()
# 	for rank in range(size):
# 		p = mp.Process(target=test_init_process, 
# 						args=(rank, size, test_run))
# 		p.start()
# 		processes.append(p)
# 	# + + +
# 	probe_cpu = ProbeCPU(pids=[p.pid for p in processes], 
# 						ranks=[rank for rank in range(size)])
# 	probe_cpu.run(processes[0])
# 	# + + +
# 	for p in processes:
# 		p.join()
