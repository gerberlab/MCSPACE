from model import MapseqTopicModel 
from configparser import ConfigParser
# from multiprocessing import Pool


def run_from_config_file(config_file):
    model = MapseqTopicModel(config_file)

    model.load_data() 
    model.init_params_mcmc()
    model.train()


if __name__ == "__main__":
    # config_file = "./quick_test.cfg"
    # config_file = "./configs/run_2comm_overlap.cfg"
    config_file = "./configs/three_comm_one_uniform.cfg"
    run_from_config_file(config_file) 


    # parser = ConfigParser()
    # parser.read(config_file)
    # n_chains = parser.getint('general', 'n_chains')

    # if n_chains > 1:
    #     with Pool(8):
    #         run_from_config_file(config_file, seed)
    # else:
    #     run_from_config_file(config_file)