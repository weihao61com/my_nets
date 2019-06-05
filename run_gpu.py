import sys
from utils import Utils


HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

model = 'r2'
config = 'rnn_config.json'
machine = 'weihao@debian-sensors'
#machine = 'weihao@sensors-debian2'

if len(sys.argv)>1 and sys.argv[1]=='1':
    machine = 'weihao@sensors-debian2'
if len(sys.argv)>2:
    model = sys.argv[2]

cmd = 'rm -r {}/NNs/{}'.format(HOME, model)
Utils.run_cmd(cmd)

cmd = 'scp -r {0}:/home/weihao/Projects/NNs/{1} {2}/NNs/{1}'.format(machine, model, HOME)
Utils.run_cmd(cmd)

cmd = 'rm -r {}/NNs/{}_fc_avg.p'.format(HOME, model)
Utils.run_cmd(cmd)

cmd = 'scp -r {0}:/home/weihao/Projects/NNs/{1}_fc_avg.p {2}/NNs/'.format(machine, model, HOME)
Utils.run_cmd(cmd)

cmd = 'rm {}/my_nets/fc/{}'.format(HOME, config)
Utils.run_cmd(cmd)

cmd = 'scp -r {0}:/home/weihao/Projects/my_nets/fc/{1} {2}/my_nets/fc/{1}'.format(machine, config, HOME)
Utils.run_cmd(cmd)
