from os import terminal_size

import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import json
import pickle


# default target vocab: /work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/nn-vocabs/vocab.de-en.en.pkl
# default source vocab: /work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/nn-vocabs/vocab.de-en.de.pkl

def main(args):

    # Get dictionaries
    with open(args.target_vocab_file, 'rb') as w:
        target_dictionary = pickle.load(w)
    target_int_to_vocab = {target_dictionary[w]: w for w in target_dictionary.keys()}

    with open(args.source_vocab_file, 'rb') as w:
        source_dictionary = pickle.load(w)
    source_int_to_vocab = {source_dictionary[w]: w for w in source_dictionary.keys()}

    # Visualize
    d = np.load(args.attention).item()
    d = [v for (k, v) in d.items()]
    print(list(d[args.t].keys()))
    l = ""
    for k in list(d[args.t].keys()):
        if len(k) > len("rec_"):
            if k[:len("rec_")] == "rec_":
                l = k
                break
    att_weights = d[args.t][l]  # TODO: assuming only 1 layer
    target = [target_int_to_vocab[w] for w in d[args.t]['classes']]
    source = [source_int_to_vocab[w] for w in d[args.t]['data']]
    
    att_weights = np.average(att_weights, axis=-1)
    #att_weights = np.squeeze(att_weights, axis=-1)
    np.set_printoptions(suppress=True)


    fig, ax = plt.subplots()
    ax.matshow(att_weights, cmap=plt.cm.Blues, aspect=0.5)
    ax.set_xticks(np.arange(len(source)))
    ax.set_yticks(np.arange(len(target)))

    fig.tight_layout()

    ax.set_xticklabels(source, size=20)
    ax.set_yticklabels(target, size=20)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    if args.show_labels:
        for i in range(len(target)):
            for j in range(len(source)):
                text = ax.text(j, i, '{0:.2f}'.format(att_weights[i, j]).rstrip("0"),
                               ha="center", va="center", color="black")

    #fig.subplots_adjust(top=0.8, left=0.1)
    if args.save_fig is None:
        plt.show()
    else:
        plt.savefig(args.save_fig, bbox_inches="tight")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('attention', metavar='attention', type=str, help='path to attention file')
    parser.add_argument('t', metavar='t', type=int, help='time step to visualize')

    # '/work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/nn-vocabs/vocab.de-en.en.pkl'
    #d_t = "/home/nikita/NMT/visualizations/attention_weights/vocab.de-en.en.pkl"
    d_t = "/u/bahar/workspace/wmt/2018/de-en-6M--2019-01-16/de-en-hmm--2018-01-16/dataset/target.vocab.pkl"

    # /work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/nn-vocabs/vocab.de-en.de.pkl
    #d_s = "/home/nikita/NMT/visualizations/attention_weights/vocab.de-en.de.pkl"
    d_s = "/u/bahar/workspace/wmt/2018/de-en-6M--2019-01-16/de-en-hmm--2018-01-16/dataset/source.vocab.pkl"

    parser.add_argument('--target_vocab_file', metavar='target_vocab_file', type=str,
                        help='Path to vocab pickle file of targets',
                        default=d_t,
                        required=False)
    parser.add_argument('--source_vocab_file', metavar='source_vocab_file', type=str,
                        help='Path to vocab pickle file of source',
                        default=d_s,
                        required=False)

    parser.add_argument('--show_labels', dest='show_labels', action='store_true')
    parser.add_argument('--save_fig', metavar='save_fig', type=str,
                        help='Path to save figure',
                        default=None,
                        required=False)

    args = parser.parse_args()
    main(args)
