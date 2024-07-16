import argparse
from BioTransformer import main as bioTrans
from BioTransformer.Corpus import TranslationCorpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inferring subclone populations")

    parser.add_argument("-dp", "--data_path", type=str,
                        help="The path for storing mutation SNV information, which requires the use of .npy format files.")
    parser.add_argument("-gt","--groundTruth",type=str,
                        help="The folder path for ground truth")
    parser.add_argument("-ed","--encode_path",
                        help="The path to load the weights of the encoding model.")
    parser.add_argument("-wp", "--weight_path", type=str,
                        help="The path to save trained weights.",default='./')
    parser.add_argument("-m", "--mode", type=str,help="Running mode, optional 'train' or 'predict'")
    args = parser.parse_args()
    if args.mode == 'train':
        data_path = args.data_path
        encode_path= args.encode_path
        weight_path = args.weight_path
        groundTruth = args.groundTruth
        corpus = TranslationCorpus(encode_path)
        enc_inputs, dec_inputs, target_batch = corpus.make_Tensor(data_path,groundTruth) # 创建训练数据
        del corpus
        bioTrans.train(enc_inputs, dec_inputs, target_batch,weight_path)
    else:
        bioTrans.predict(args.data_path,args.encode_path,args.groundTruth)