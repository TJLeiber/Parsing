import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, BertModel
import torch
import torch.nn as nn
import argparse
from BiaffineGraphBasedParser.Model import *
from Data.Preprocessing import *
from Data.Embeddings import *


class Parser:
    def __init__(self, model: str = "default", device: str = 'cpu'):
        self.device = torch.device('cuda' if (device == 'gpu' and torch.cuda.is_available()) else 'cpu')
        if device == 'gpu' and self.device == torch.device('cpu'):
            print("No GPU found. Moving on to CPU.")

        if model == "fast":
            model_path = "../models/model_static.pth"
        elif model == "default":
            model_path = "../models/model_concat.pth"
        else:
            raise ValueError(f"Invalid input for kwarg 'model'. Must chose either 'fast' or 'default' not {model}.")
        try:
            self.model = torch.load(model_path, map_location=self.device)
            assert isinstance(self.model, GraphBasedParser), "Loaded model is not of type GraphBasedParser."
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def parse(self, sentences: list, is_split_into_words: bool = False, to_tensor: bool = True, to_file: str = None, to_df: bool = False):
        if not (to_tensor or to_df or to_file):
            raise ValueError("Must specify at least one output type: to_tensor, to_df, or to_file")

        sentences_ = [["<ROOT>"] + (word_tokenize(sent) if not is_split_into_words else sent) for sent in sentences]
        pred = self.model.predict(sentences_)  # Assuming predict method exists on GraphBasedParser

        out = {}
        if to_tensor:
            out["output_tensor"] = pred.detach().cpu().numpy()

        if to_df:
            df_lst = [
                pd.DataFrame(pred[i].numpy()[:len(sent), 1:len(sent)],
                             index=[f"{word}-head" for word in sent],
                             columns=[f"{word}-dep" for word in sent[1:]])
                for i, sent in enumerate(sentences_)
            ]
            out["dataframes"] = df_lst

        if to_file:
            df_lst = df_lst if to_df else [
                pd.DataFrame(pred[i].numpy()[:len(sent), 1:len(sent)],
                             index=[f"{word}-head" for word in sent],
                             columns=[f"{word}-dep" for word in sent[1:]])
                for i, sent in enumerate(sentences_)
            ]
            with open(to_file, 'a') as f:
                for i, df in enumerate(df_lst, start=1):
                    f.write(f"#{i}\n{df.to_string()}\n\n")

        return out


def main():
    argparser = argparse.ArgumentParser(
        prog="Biaffine Graph Based Parser",
        description="Biaffine Graph Based Parser to parse sentences in 'source_file' and write to 'target_file'"
    )
    # arguemnts concerning the content to be parsed
    argparser.add_argument("source_file", nargs="?", default=None, type=str,
                           help="Path to file containing sentences that need to be parsed. Input of a source file is never considered to be tokenized.")
    argparser.add_argument("target_file", nargs="?", default=None, type=str,
                           help="Path to file to which parsed sentences will be written. If file already exists, parsed sentences will be appended.")

    argparser.add_argument("-c", "--concat", action="store_true",
                           help="Use model with concatenated BERT and GloVe embeddings. Otherwise, only GloVe. (transformers version 4.41.2 recommended)")
    argparser.add_argument("-g", "--gpu", action="store_true",
                           help="Use GPU for inference if available. Otherwise, use CPU.")
    argparser.add_argument("-i", "--individual_sentences", action="store_true",
                           help="Each line in the source file contains an individual sentence.")
    args = argparser.parse_args()

    device = "gpu" if args.gpu else "cpu"
    embedding_type = "default" if args.concat else "fast"
    parser = Parser(model=embedding_type, device=device) # load appropriate model


    assert args.source_file is not None, "Please provide a source file as input to be parsed. For help pass 'python main.py -h'."
    with open(args.source_file, "r") as f:
        if args.individual_sentences:
            sentences = [sent.rstrip("\n") for sent in f.readlines()] # given a structured file with each line being a sentence
        else:
            sentences = sent_tokenize(f.read(), language="english") # given an entire text

        parser.parse(sentences, is_split_into_words=False, to_tensor=False, to_df=False, to_file=args.target_file)


if __name__ == "__main__":
    main()
