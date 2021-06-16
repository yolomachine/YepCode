from __future__ import annotations

import argparse
import os.path
import time
from typing import List, Tuple
from java_ast_provider import JavaST, ParsedNode
from tqdm import tqdm


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
    def sass(output_dir_path: str, paths: List[str] = None, source_codes: List[str] = None) -> None:
        try:
            corpus, vocab = JavaSTEncoder.sequentialize(paths=paths, source_codes=source_codes)
            if len(corpus) > 0 and len(vocab) > 0:
                corpus_path = os.path.join(output_dir_path, 'java.corpus')
                vocab_path = os.path.join(output_dir_path, 'java.vocab')

                with open(corpus_path, 'w', encoding='utf-8') as fp:
                    print('\n'.join(corpus), file=fp)
                print(f'Generated {corpus_path}')

                with open(vocab_path, 'w', encoding='utf-8') as fp:
                    print('\n'.join(vocab), file=fp)
                print(f'Generated {vocab_path}')
        except:
            print(f'Couldn\'t generate corpus and vocabulary')


def generate_sequence(args) -> None:
    path = args.path
    JavaSTEncoder.sass(path, paths=[path])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Sequentialize Java AST.')
    argparser.add_argument('path',
                           help='path to .java file',
                           action='store')
    generate_sequence(argparser.parse_args())
