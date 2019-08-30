import sys
from utils import Utils


HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

model = 'kitty6'
config = 'config.json'
config_save = 'config_save.json'
machine = 'weihao@debian-sensors'
sub_dir = 'hidden_f'
#machine = 'weihao@sensors-debian2'

if len(sys.argv)>1 and sys.argv[1]=='1':
    machine = 'weihao@sensors-debian2'
if len(sys.argv)>2:
    model = sys.argv[2]

cmd = 'rm -r {}/NNs/{}'.format(HOME, model)
Utils.run_cmd(cmd)

cmd = 'scp -r {0}:/home/weihao/Projects/NNs/{1} {2}/NNs/{1}'.format(machine, model, HOME)
Utils.run_cmd(cmd)

cmd = 'cp {}/NNs/{}_fc_avg.p {}/NNs/{}_save_fc_avg.p'.format(HOME, model, HOME, model)
Utils.run_cmd(cmd)

cmd = 'scp -r {0}:/home/weihao/Projects/NNs/{1}_fc_avg.p {2}/NNs/'.format(machine, model, HOME)
Utils.run_cmd(cmd)

cmd = 'cp {}/my_nets/fc/{} {}/my_nets/{}/{}'.format(HOME, config, HOME, sub_dir, config_save)
Utils.run_cmd(cmd)

cmd = 'scp -r {0}:/home/weihao/Projects/my_nets/{3}/{1} {2}/my_nets/{3}/{1}'.format(machine, config, HOME, sub_dir)
Utils.run_cmd(cmd)
