from __future__ import annotations

import argparse
import os.path
import time
import multiprocessing
from typing import List, Tuple
from java_ast_provider import JavaST, ParsedNode
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim import utils


class Corpus:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, 'r', encoding='utf-8') as fp:
            for line in fp:
                yield utils.simple_preprocess(line)


class JavaSTEncoder:
    @staticmethod
    def __unfold_node(node: ParsedNode, sep=' '):
        output = repr(node)
        for c in node.children:
            output = f'{output}{sep}{JavaSTEncoder.__unfold_node(c, sep)}'
        return output

    @staticmethod
    def __unfold_tree(tree: JavaST, sep=' '):
        return f'{JavaSTEncoder.__unfold_node(tree.root, sep)}'

    @staticmethod
    def read(path, progress_bar: tqdm = None):
        output = []
        if os.path.isdir(path):
            for content in os.listdir(path):
                output += JavaSTEncoder.read(os.path.join(path, content), progress_bar)
        else:
            if os.path.splitext(path)[1] == '.java':
                with open(path, 'r', encoding='utf-8') as fp:
                    output.append(fp.read())
                if not isinstance(progress_bar, type(None)):
                    progress_bar.update()
        return output

    @staticmethod
    def sequentialize(paths: List[str] = None, source_codes: List[str] = None) -> Tuple[List[str], List[str]]:
        source_codes = source_codes or []
        paths = paths or []
        progress_bar = tqdm(position=0,
                            leave=True,
                            desc='Reading source codes')
        for p in paths:
            try:
                for sc in JavaSTEncoder.read(p, progress_bar):
                    source_codes.append(sc)
            except:
                pass
            finally:
                progress_bar.close()
                time.sleep(0.5)
        corpus = set()
        vocabulary = set()
        for sc in tqdm(source_codes,
                       total=len(source_codes),
                       position=0,
                       leave=True,
                       desc=f'Building corpus and vocabulary'):
            try:
                tree = JavaST(source_code=sc)
                sequences = list(map(lambda s: JavaSTEncoder.__unfold_tree(s), tree.as_statement_tree_sequence()))
                for s in sequences:
                    corpus.add(s)
                    for w in s.split(' '):
                        vocabulary.add(w)
            except:
                pass
        return list(corpus), list(vocabulary)

    @staticmethod
    def sass(output_dir: str, paths: List[str] = None, source_codes: List[str] = None) -> str:
        try:
            corpus, vocab = JavaSTEncoder.sequentialize(paths=paths, source_codes=source_codes)
            if len(corpus) > 0 and len(vocab) > 0:
                corpus_path = os.path.join(output_dir, 'java.corpus')

                with open(corpus_path, 'w', encoding='utf-8') as fp:
                    print('\n'.join(corpus), file=fp)
                print(f'Generated {corpus_path}')

                return corpus_path
        except:
            print(f'Couldn\'t generate corpus')

    @staticmethod
    def w2v(corpus_path: str) -> Word2Vec:
        cores = multiprocessing.cpu_count()
        w2v_model = Word2Vec(sentences=Corpus(corpus_path),
                             workers=cores-1, min_count=1,
                             epochs=30, vector_size=192, sample=1e-3)
        return w2v_model


def generate_corpus(path: str, output_dir: str = None) -> None:
    output_dir = output_dir or (path if os.path.isdir(path) else os.path.dirname(path))
    JavaSTEncoder.sass(output_dir, paths=[path])


def learn_w2v_embeddings(path: str, output_dir: str = None):
    output_dir = output_dir or (path if os.path.isdir(path) else os.path.dirname(path))
    w2v_model = JavaSTEncoder.w2v(path)
    name = 'java.w2v'
    if not os.path.isdir(path):
        pre, ext = os.path.splitext(path)
        name = f'{pre}{ext}.w2v'
    model_path = os.path.join(output_dir, name)
    try:
        w2v_model.save(fname_or_handle=model_path)
        print(f'Generated {model_path}')
    except:
        print(f'Couldn\'t save Word2Vec model')


def load_w2v_model(path: str):
    try:
        w2v_model = Word2Vec.load(fname=path)
        pre, ext = os.path.splitext(os.path.basename(path))
        sim_path = os.path.join(os.path.dirname(path), pre + ext + '.sim')
        print(w2v_model.wv.index_to_key)
        print(f'Total words: {len(w2v_model.wv.index_to_key)}')
        with open(sim_path, 'w', encoding='utf-8') as fp:
            for i, w in enumerate(w2v_model.wv.index_to_key):
                sim = "\n".join(map(str, w2v_model.wv.most_similar(positive=[w])))
                print(f'{w}[{sim}]\n', file=fp)
        print(f'Generated {sim_path}')
    except:
        print(f'Couldn\'t load {path}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Sequentialize Java AST.')
    argparser.add_argument('-jp', '--java', metavar='PATH',
                           help='path to `.java` file or directory containing `.java` files, generates corpus to [-od]',
                           action='store')
    argparser.add_argument('-cp', '--corpus', metavar='PATH',
                           help='path to `.corpus` file to learn embeddings from',
                           action='store')
    argparser.add_argument('-od', '--output', metavar='PATH',
                           help='path to output directory for [-jp, --java], [-w2v], [-w2v_sim]',
                           action='store')
    argparser.add_argument('-w2v',
                           help='Word2Vec',
                           action='store_const',
                           const=True,
                           default=False)
    argparser.add_argument('-ft',
                           help='fastText',
                           action='store_const',
                           const=True,
                           default=False)
    argparser.add_argument('-w2v_sim', metavar='PATH',
                           help='path to Word2Vec model to generate similarities from',
                           action='store')

    args = argparser.parse_args()
    if args.java:
        generate_corpus(args.java, output_dir=args.output)

    if args.corpus:
        if args.w2v:
            learn_w2v_embeddings(args.corpus, output_dir=args.output)\

    if args.w2v_sim:
        load_w2v_model(args.w2v_sim)
