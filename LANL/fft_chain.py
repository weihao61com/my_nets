from train_fft import main1
from L_fc import main2
from train_fft_test import main3
from train_fft_refit import main4

config = 'config.json'

main1(config, 10)
main2(config, True)
main3(config)
main4()

