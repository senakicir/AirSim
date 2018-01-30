import os
import os.path
import numpy
import numpy as np

import torch

import logging

import IPython
import logging


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def config_logger(log_file="/dev/null", level=logging.INFO):
    class MyFormatter(logging.Formatter):
        info_format = "\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s"
        error_format = "\x1b[31;1m%(asctime)s [%(name)s] [%(levelname)s]\x1b[0m %(message)s"

        def format(self, record):
            if record.levelno > logging.INFO:
                self._style._fmt = self.error_format
            else:
                self._style._fmt = self.info_format

            res = super(MyFormatter, self).format(record)
            return res

    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_file)
    fileFormatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s]> %(message)s")
    fileHandler.setFormatter(fileFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleFormatter = MyFormatter()
    consoleHandler.setFormatter(consoleFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(level)

def transfer_partial_weights(state_dict_other, obj, submodule=0, prefix=None):
    print('Transferring weights...')

    # print('\nStates source\n')
    # for name, param in state_dict_other.items():
    #     print(name)
    # print('\nStates target\n')
    # for name, param in obj.state_dict().items():
    #     print(name)
    
    own_state = obj.state_dict()
    copyCount = 0
    skipCount = 0
    paramCount = len(own_state)
    #for name_raw, param in own_state.items():
    #    paramCount += param.view(-1).size()[0]
    #for name_raw, param in state_dict_other.items():
    #    print("param",param)

    for name_raw, param in state_dict_other.items():
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
            #print('.data conversion for ',name)
        if prefix is not None and not name_raw.startswith(prefix):
            #print("skipping {} because of prefix {}".format(name_raw, prefix))
            continue
        
        # remove the path of the submodule from which we load
        name = ".".join(name_raw.split('.')[submodule:])

        if name in own_state:
            if hasattr(own_state[name],'copy_'): #isinstance(own_state[name], torch.Tensor):
                #print('copy_ ',name)
                if own_state[name].size() == param.size():
                    own_state[name].copy_(param)
                    copyCount += 1
                else:
                    print('Invalid param size(own={} vs. source={}), skipping {}'.format(own_state[name].size(), param.size(), name))
                    skipCount += 1
            
            elif hasattr(own_state[name],'copy'):
                own_state[name] = param.copy()
                copyCount += 1
            else:
                print('training.utils: Warning, unhandled element type for name={}, name_raw={}'.format(name,name_raw))
                print(type(own_state[name]))
                skipCount += 1
                IPython.embed()
        else:
            skipCount += 1
            print('Warning, no match for {}, ignoring'.format(name))
            #print(' since own_state.keys() = ',own_state.keys())
    print('Copied {} elements, {} skipped, and {} target params without source'.format(copyCount, skipCount, paramCount-copyCount))
