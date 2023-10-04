from utils.config import Config
from run.AVMNIST_main import AVMNIST_main
from run.CREMAD_main import CREMAD_main
from run.URFunny_main import URFunny_main
from run.AVE_main import AVE_main
from run.MOSEI_main import MOSEI_main
from eval.MOSEI_eval import MOSEI_eval

def main():
    cfgs = Config()
    
    if cfgs.mode == 'train':
        if cfgs.dataset == "AV-MNIST":
            AVMNIST_main(cfgs)
        elif cfgs.dataset == "CREMAD":
            CREMAD_main(cfgs)
        elif cfgs.dataset == "URFunny":
            URFunny_main(cfgs)
        elif cfgs.dataset == "AVE":
            AVE_main(cfgs)
        elif cfgs.dataset == "MOSEI":
            MOSEI_main(cfgs)
        elif cfgs.dataset == 'MOSEI-EMO':
            MOSEI_main(cfgs)
    else:
        if cfgs.dataset == "MOSEI":
            MOSEI_eval(cfgs)
        elif cfgs.dataset == 'MOSEI-EMO':
            MOSEI_eval(cfgs)
    
    


if __name__ == '__main__':
    main()
