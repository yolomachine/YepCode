import os
import re
import json
import tqdm
import subprocess
import numpy as np
import clang.cindex
import clang.enumerations

from tokens import *
from multiprocessing.dummy import Pool as ThreadPool

def get_identifier(identifiers, token, m=max_identifiers, name='IDENTIFIER'):
    if token.spelling in identifiers:
        return identifiers[token.spelling]
    else:
        if len(identifiers) < m:
            identifiers[token.spelling] = (name := f'{name}_{len(identifiers) + 1}')
        return name

def parse_template_type(types, identifiers, tokens):
    yield from parse_type(types, identifiers, tokens, True)
    
    token = tokens.pop()
    while token.spelling in {',', '...'}:
        yield punctuation[token.spelling]
        
        if tokens[-1].spelling != '>':
            yield from parse_type(types, identifiers, tokens, True)
            
        token = tokens.pop()
        
    if token.spelling == '>':
        yield 'GREATER'
    else:
        tokens.append(token)

def parse_type(types, identifiers, tokens, is_tmpl=False):
    token = tokens.pop()
    
    if token.spelling == '[':
        while token.spelling == '[':
            yield punctuation[token.spelling]
            while tokens[-1].spelling != ']':
                yield from parse_token(types, identifiers, tokens)
            yield punctuation[tokens.pop().spelling]

            token = tokens.pop()

        tokens.append(token)
        return
    
    while token.kind == clang.cindex.TokenKind.KEYWORD:
        yield keywords.get(token.spelling, 'KEYWORD')
        token = tokens.pop()
    
    if token.spelling == 'std':
        colon = tokens.pop()
        if colon.spelling == '::':
            token = tokens.pop()
            name = f'std::{token.spelling}'
            if name in special_identifiers:
                yield special_identifiers[name]
            else:
                yield 'STD'
                yield get_identifier(identifiers, token)
            token = tokens.pop()
        else:
            yield 'STD'
            token = colon
                    
    if token.spelling != '<':
        name = token.spelling
        
        if name in special_types:
            yield special_types[name]
            return
    
        if name not in types:
            if is_tmpl:
                name = get_identifier(types, token, m=max_types, name='TYPE')
            else:
                tokens.append(token)
                return

        while isinstance(name, str) and name in types:
            name = types[name]
        
        if isinstance(name, list):
            yield from name
        else:
            yield name
        
        token = tokens.pop()
        
    while token.spelling == '[':
        yield punctuation[token.spelling]
        while tokens[-1].spelling != ']':
            yield from parse_token(types, identifiers, tokens)
        yield punctuation[tokens.pop().spelling]

        token = tokens.pop()
        
    if token.spelling == '<':
        yield 'LESS'
        yield from parse_template_type(types, identifiers, tokens)
    else:
        tokens.append(token)

def parse_typedef(types, identifiers, tokens):
    t = list(parse_type(types, identifiers, tokens))
    name = tokens.pop()
    tokens.pop() # eat semicolon
    
    types[name.spelling] = t

def parse_using(types, identifiers, tokens):
    name = tokens.pop()
    tokens.pop() # eat =
    t = list(parse_type(types, identifiers, tokens, True))
    while tokens[-1].spelling != ';':
        t.extend(parse_token(types, identifiers, tokens))
    
    token = tokens.pop() # eat semicolon
    
    types[name.spelling] = t

def parse_token(types, identifiers, tokens):
    token = tokens.pop()    
        
    if token.kind == clang.cindex.TokenKind.COMMENT:
        return
    
    if token.kind == clang.cindex.TokenKind.PUNCTUATION:
        if token.spelling == '#':
            token = tokens.pop()
            if token.spelling == 'include':
                token = tokens.pop() # < or "
                if token.spelling == '<':
                    name = ''
                    token = tokens.pop()
                    while token.spelling != '>':
                        name += token.spelling
                        token = tokens.pop()
                elif token.spelling == '"':
                    name = ''
                    token = tokens.pop()
                    while token.spelling != '"':
                        name += token.spelling
                        token = tokens.pop()
                else:
                    name = token.spelling[1:-1]
                
                yield includes.get(name, 'INCLUDE');
            return

        name = token.spelling

        if '&' in name:
            if token.cursor.kind == clang.cindex.CursorKind.VAR_DECL:
                name += 'D'
            elif token.cursor.kind == clang.cindex.CursorKind.UNARY_OPERATOR:
                name += 'U'
            elif token.cursor.kind == clang.cindex.CursorKind.PARM_DECL or token.cursor.kind == clang.cindex.CursorKind.FUNCTION_TEMPLATE:
                name += 'P'
            elif token.cursor.kind == clang.cindex.CursorKind.LAMBDA_EXPR:
                name += 'L'

        yield punctuation.get(name, 'PUNCTUATION')
        return
    
    if token.kind == clang.cindex.TokenKind.LITERAL:        
        key = token.cursor.kind if token.cursor.kind in literals else token.spelling        
        yield literals.get(key, 'LITERAL')
        return

    if token.kind == clang.cindex.TokenKind.KEYWORD:
        if token.spelling == 'typedef':
            parse_typedef(types, identifiers, tokens)
            return
        
        if token.spelling == 'using' and token.cursor.kind == clang.cindex.CursorKind.TYPE_ALIAS_DECL:
            parse_using(types, identifiers, tokens)
            return
        
        if token.cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            tokens.append(token)
            yield from parse_type(types, identifiers, tokens)
            token = tokens.pop()
            if token.spelling == 'main':
                yield 'MAIN_FUNCTION'
            else:
                tokens.append(token)
            return
            
        if token.cursor.kind in { clang.cindex.CursorKind.TYPE_REF, clang.cindex.CursorKind.VAR_DECL, clang.cindex.CursorKind.PARM_DECL, clang.cindex.CursorKind.DECL_STMT }:              
            tokens.append(token)
            yield from parse_type(types, identifiers, tokens)
            return
        
        if token.cursor.kind not in { clang.cindex.CursorKind.TEMPLATE_TYPE_PARAMETER, clang.cindex.CursorKind.TEMPLATE_TEMPLATE_PARAMETER } and token.spelling in { 'class', 'struct' }:
            name = token.spelling.upper()
            yield keywords.get(token.spelling, 'KEYWORD')
            token = tokens.pop()
            if token.kind == clang.cindex.TokenKind.IDENTIFIER:
                yield get_identifier(types, token, m=max_types, name='TYPE')
                return

            tokens.append(token)
            return
                
        yield keywords.get(token.spelling, 'KEYWORD')
        return

    if token.kind == clang.cindex.TokenKind.IDENTIFIER:
        if token.cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            if token.spelling == 'main':
                yield 'MAIN_FUNCTION'
                return
        
        if token.cursor.kind in { clang.cindex.CursorKind.TYPE_REF, clang.cindex.CursorKind.VAR_DECL, clang.cindex.CursorKind.PARM_DECL, clang.cindex.CursorKind.DECL_STMT }:
            tokens.append(token)
            result = list(parse_type(types, identifiers, tokens))
            if result:
                yield from result
            else:
                tokens.pop()
                if token.cursor.kind in decl_types:
                    yield decl_types[token.cursor.kind]
                    
                if token.spelling in special_identifiers:
                    yield special_identifiers[token.spelling]
                else:
                    yield get_identifier(identifiers, token)
            return
        
        if token.spelling == 'std':
            colon = tokens.pop()
            if colon.spelling == '::':
                token = tokens.pop()
                name = f'std::{token.spelling}'
                if name in special_identifiers:
                    yield special_identifiers[name]
                else:
                    yield 'STD'
                    yield get_identifier(identifiers, token)
            else:
                yield 'STD'
                tokens.append(colon)
                
            return

        if token.spelling in special_identifiers:
            yield special_identifiers[token.spelling]
            return
        
        if token.spelling in types:
            data = types[token.spelling]
            if isinstance(data, list):
                yield from data
            else:
                yield data
            return

        name = get_identifier(identifiers, token)

        if token.cursor.kind in decl_types:
            yield decl_types[token.cursor.kind]

        yield name

clangindex = clang.cindex.Index.create()

def parse_tokens_path(path):
    try:
        tu = clangindex.parse(path)
        tokens = list(tu.cursor.get_tokens())[::-1]
    except:        
        return

    identifiers = {}
    types = {}
    
    while tokens:
        yield from parse_token(types, identifiers, tokens)

def parse_tokens_str(source, id):
    print(source, file=open(f'tmp/{id}.cpp', 'w'))
    yield from parse_tokens_path(f'tmp/{id}.cpp')

reg_include = re.compile(r'#include ?[<"].+?[>"]')
reg_sharp = re.compile(r'^# .*$', flags=re.MULTILINE)
reg_clean = re.compile(r'using namespace std;')


def read_data(path):
    filepath = path + '.cpp'
    if os.path.exists(filepath):
        return open(filepath).read()

    filepath = path + '.json'
    if os.path.exists(filepath):
        return json.load(open(filepath))['Source-Code']

    if os.path.exists(path):
        return json.load(open(path))['Source-Code']
    
    return None


def process_one(subfolder, id):
    path = os.path.join(subfolder, 'cpp', id)

    data = read_data(path)
    if data is None:
        return None

    os.makedirs(f'tmp/{subfolder}/', exist_ok=True)
        
    filename = f'tmp/{subfolder}/{id}.cpp'
    includes = reg_include.findall(data)
    with open(filename, 'w') as file:
        print(reg_include.sub('', data), file=file)
    
    cmd = ['g++', '-E', filename]; 
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    
    if exit_code != 0:
        return None    
    
    try:
        output = output.decode('utf-8')
    except:
        return None

    output = reg_sharp.sub('', output)
    output = reg_clean.sub('', output)
    
    with open(filename, 'w') as file:
        print('\n'.join(includes) + output, file=file)
        
    tokens = [token_to_id[i] + 1 for i in parse_tokens_path(filename) if isinstance(i, str) and i not in args.skip_tokens]
    if not tokens:
        return None
        
    return tokens


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', help='Path to dataset')
parser.add_argument('list_file', help='Path to list file')
parser.add_argument('output_file', help='Path to output file')
parser.add_argument('--skip_tokens', help='Tokens to replace skip', nargs='*', default=set())
args = parser.parse_args()

args.skip_tokens = set(args.skip_tokens)

output_file = open(args.output_file, 'w')

pbar = tqdm.tqdm(open(args.list_file).readlines())
for i in pbar:
    i = i.strip()
    if not i:
        continue

    index, id, *tags = i.split()

    if not tags:
        continue

    pbar.set_description(f'{index} {id}')    

    tkns = process_one(os.path.join(args.dataset_path, index), id)
    if tkns:
        print(index, id, ' '.join(str(t) for t in tkns), ' '.join(tags), sep='\t', file=output_file)
