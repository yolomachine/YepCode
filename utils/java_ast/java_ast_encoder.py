from __future__ import annotations

import argparse
import os.path
import time
import multiprocessing
import re
from typing import List, Union
from java_ast_provider import JavaST, ParsedNode
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models import FastText


class Corpus:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, 'r', encoding='utf-8') as fp:
            for line in fp:
                yield re.findall(r"[^\s\"\']+|\"[^\"]*\"|\'[^\']*\'", line)


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
    def sequentialize(paths: List[str] = None, source_codes: List[str] = None, output_dir: str = None, prune: bool = False) -> List[str]:
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
        for sc in tqdm(source_codes,
                       total=len(source_codes),
                       position=0,
                       leave=True,
                       desc=f'Building corpus'):
            try:
                tree = JavaST(source_code=sc)
                sequences = list(map(lambda s: JavaSTEncoder.__unfold_tree(s), tree.as_statement_tree_sequence(prune)))
                for s in sequences:
                    corpus.add(s)
            except:
                pass

        corpus = list(corpus)
        if output_dir:
            try:
                corpus_path = os.path.join(output_dir, 'java.corpus')
                with open(corpus_path, 'w', encoding='utf-8') as fp:
                    print('\n'.join(corpus), file=fp)
                print(f'Generated {corpus_path}')
            except:
                print(f'Couldn\'t serialize corpus')

        return corpus

    @staticmethod
    def train_model(model: str, corpus_path: str) -> Union[Word2Vec, FastText]:
        path = os.path.join(os.path.dirname(corpus_path),
                            os.path.splitext(os.path.basename(corpus_path))[0] + f'.{model}')
        sims_path = path + '.sims'
        cores = multiprocessing.cpu_count()

        if model == 'w2v':
            print('Training Word2Vec')
            w2v_model = Word2Vec(sentences=Corpus(corpus_path),
                                 workers=cores - 1, min_count=1,
                                 epochs=30, vector_size=192, sample=1e-3)
            try:
                w2v_model.save(fname_or_handle=path)
                print(f'Generated {path}')
            except Exception as e:
                print(e)
                print(f'Couldn\'t save Word2Vec model')

            try:
                with open(sims_path, 'w', encoding='utf-8') as fp:
                    for i, w in enumerate(w2v_model.wv.index_to_key):
                        sim = "\n".join(map(str, w2v_model.wv.most_similar(positive=[w])))
                        print(f'{w}[{sim}]\n', file=fp)
                print(f'Generated {sims_path}')
            except Exception as e:
                print(e)
                print(f'Couldn\'t save Word2Vec model similarities')

            return w2v_model

        if model == 'ft':
            print('Training fastText')
            ft_model = FastText(sentences=Corpus(corpus_path),
                                workers=cores - 1, min_count=1,
                                epochs=30, vector_size=192, sample=1e-3)
            try:
                ft_model.save(fname_or_handle=path)
                print(f'Generated {path}')
            except Exception as e:
                print(e)
                print(f'Couldn\'t save fastText model')

            try:
                with open(sims_path, 'w', encoding='utf-8') as fp:
                    for i, w in enumerate(ft_model.wv.index_to_key):
                        sim = "\n".join(map(str, ft_model.wv.most_similar(positive=[w])))
                        print(f'{w}[{sim}]\n', file=fp)
                print(f'Generated {sims_path}')
            except Exception as e:
                print(e)
                print(f'Couldn\'t save fastText model similarities')

            return ft_model


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Sequentialize Java AST.')
    argparser.add_argument('-od', '--output', metavar='PATH',
                           help='path to output directory for [-jp, --java]',
                           action='store')
    argparser.add_argument('-jp', '--java', metavar='PATH',
                           help='path to `.java` file or directory containing `.java` files, '
                                'generates corpus to given path or [-od] if provided ',
                           action='store')
    argparser.add_argument('-cp', '--corpus', metavar='PATH',
                           help='path to `.corpus` file to learn embeddings from',
                           action='store')
    argparser.add_argument('-prune',
                           help='prunes the AST if provided',
                           action='store_const',
                           const=True,
                           default=False)
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

    args = argparser.parse_args()
    if args.java:
        output_dir = args.output or (args.java if os.path.isdir(args.java) else os.path.dirname(args.java))
        JavaSTEncoder.sequentialize(paths=[args.java], output_dir=output_dir, prune=args.prune)

    if args.corpus:
        if args.w2v:
            JavaSTEncoder.train_model(model='w2v', corpus_path=args.corpus)
        if args.ft:
            JavaSTEncoder.train_model(model='ft', corpus_path=args.corpus)
