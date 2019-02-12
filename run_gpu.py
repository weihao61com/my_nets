import sys
from utils import Utils

model = 'r2'
config = 'rnn_config.json'
machine = 'weihao@debian-sensors'
machine = 'weihao@sensors-debian2'

if len(sys.argv)>1:
    machine = sys.argv[1]

cmd = 'rm -r /Users/weihao/Projects/NNs/{}'.format(model)
Utils.run_cmd(cmd)

cmd = 'scp -r {0}:/home/weihao/Projects/NNs/{1} /Users/weihao/Projects/NNs/{1}'.format(machine, model)
Utils.run_cmd(cmd)

cmd = 'rm /Users/weihao/Projects/my_nets/fc/{}'.format(config)
Utils.run_cmd(cmd)

cmd = 'scp -r {0}:/home/weihao/Projects/my_nets/fc/{1} /Users/weihao/Projects/my_nets/fc/{1}'.format(machine, config)
Utils.run_cmd(cmd)

