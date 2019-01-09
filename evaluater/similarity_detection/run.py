import argparse
import evaluater.similarity_detection.experiments as ex

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',
                        required=True,
                        type=int,
                        help='Experiment selection')

    parse_args, _ = parser.parse_known_args()
    states = {}

    if parse_args.exp == 1:
        states = ex.experiment_seq2seq_siamese()
    elif parse_args.exp == 2:
        states = ex.experiment_seq2_siamese()
    elif parse_args.exp == 3:
        states = ex.experiment_seq2seq()
    elif parse_args.exp == 4:
        states = ex.experiment_cnn_kim()
    elif parse_args.exp == 5:
        states = ex.experiment_cnn_tck()
    elif parse_args.exp == 6:
        states = ex.experiment_cnn_tck2()
    elif parse_args.exp == 7:
        states = ex.experiment_seq2seq_siamese_sdep()
    elif parse_args.exp == 8:
        states = ex.experiment_seq2_siamese_sdep()
    elif parse_args.exp == 9:
        states = ex.experiment_seq2seq_sdep()
    elif parse_args.exp == 10:
        states = ex.experiment_cnn_kim_sdep()
    elif parse_args.exp == 11:
        states = ex.experiment_cnn_tck_sdeq()
    elif parse_args.exp == 12:
        states = ex.experiment_seq2seq_sdep_2()
    elif parse_args.exp == 13:
        states = ex.experiment_seq2seq_siamese_sdep_2()
    else:
        print("Bad experiment code")
    print(states)

