from multiprocessing import Pool

import os
import os.path as osp
import subprocess
import time
import random
from copy import deepcopy
import random


def cfg2cmd(ccfg, gpuID=0):
    cfg = deepcopy(ccfg)
    cmd = "CUDA_VISIBLE_DEVICES=%d " % gpuID
    cmd += " python %s " % cfg["main"]
    del cfg["main"]
    for key, value in cfg.items():
        cmd += " --%s %s" % (key, value)
    return cmd


def run(cfg, gpuID=None):
    # time.sleep(random.randint(1, 10) * 3)
    cmd = "CUDA_VISIBLE_DEVICES=%d " % gpuID
    cmd += " python %s " % cfg["main"]
    del cfg["main"]
    for key, value in cfg.items():
        cmd += " --%s %s" % (key, value)

    os.makedirs("logs", exist_ok=True)
    with open("logs/gpu-%d.txt" % gpuID, "w+") as fp:
        subprocess.call(cmd, shell=True, stdout=fp, stderr=fp)


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='CFG',
                        help='path to configuration')
    parser.add_argument('-t', '--threads', type=int, default=4,
                        help="how many GPUs are avaliable on this machine")
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        configs = json.load(fp)
    threads = args.threads

    # if threads <= 1:
    #     run(configs[0])
    # else:
    #     p = Pool(4)
    #     p.map(run, configs)
    it = iter(configs)
    from threading import Thread
    import time
    import random
    from queue import Queue
    from threading import Thread, Condition, Lock

    my_queue = Queue(threads)


    def Producer():
        global my_queue
        for each in configs:
            my_queue.put(each)


    class Consumer():
        def __init__(self, id):
            self.id = id

        def __call__(self):
            global my_queue
            while not my_queue.empty():
                cfg = my_queue.get()
                cmd = cfg2cmd(cfg, self.id)
                print("Consumer %d:" % self.id, cmd)
                run(cfg, self.id)
                time.sleep(1)
                my_queue.task_done()


    P = Thread(name="Producer", target=Producer)

    C_list = []
    for i in range(threads):
        c = Thread(name="Consumer%d" % i, target=Consumer(i))
        C_list.append(c)

    P.start()
    for consumer in C_list:
        consumer.start()
