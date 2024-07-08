"""
Authors: GY-seeker
"""
import argparse
from VAEManager import TrainPrediction

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train TransformerVAE")

    parser.add_argument("-e", "--epoch", type=int, help="The epoch of trainning, default:200",default=200)
    parser.add_argument("-b", "--batch", type=int, help="The batch size of VAE model", default=64)
    parser.add_argument("-lr", "--learn_rate", type=float, help="the learn rate,default:0.02", default=0.02)
    parser.add_argument("-g", "--genomatic_length", type=int,
                        help="The length of gene fragments,Different based on data from WGS and WES. \n"+
                             "WGS:1000~300， WES:300~1000. \n"+
                             "default:2048 ",
                        default=2048)
    parser.add_argument("-ld", "--latent_dim", type=int,
                        help="The value between half of genomatic_length and genomatic_length. \n"+
                         "The smaller the latent_dim, the less memory is occupied and the poorer the simulation ability. \n"+
                         "The lager the latent_dim, the more memory is occupied, and the better the simulation ability. \n"+
                        "default:1024", default=1024)
    parser.add_argument("-p", "--data_path", type=str,help="The folder path for .npy files")
    parser.add_argument("-dt", "--data_type", type=str,help="'CN' (Copy Number) or 'SNV' (Single Nucleotide Variate)")
    parser.add_argument("-m", "--mode", type=str,help="Running mode, optional 'train' or 'predict'")
    args = parser.parse_args()

    TP = TrainPrediction(total_epochs = args.epoch, batch_size = args.batch,
                         lr = args.learn_rate, genomatic_length = args.genomatic_length,
                         latent_dim = args.latent_dim)
    if args.mode == 'train':
        TP.load_sample(args.data_path,args.data_type)
        TP.trainning()
    elif args.mode == 'predict':
        TP.load_sample(args.data_path, args.data_type)
        x_total,y_total = TP.prediction()
        # TO SHOW THE RESULT
        # plt.plot(x_total[],'o',linestyle='')
        # plt.plot(y_total[],'o',linestyle='')
        # plt.show()


