import json
import os
import javalang
import argparse
from collections import defaultdict


class ParsedNode:
    def __init__(self, javalang_node):
        self._decl = javalang_node
        self.type = type(javalang_node).__name__
        self.token = f''
        self.parent = None
        self.children = []

    def __repr__(self):
        return f'{self.type}' + (f' - `{self.token}`' if self.token else '')


class CompilationUnit(ParsedNode):
    def __init__(self, javalang_node):
        super().__init__(javalang_node)
        if not self._decl:
            return
        self.token = f'{self._decl.package}'


class Import(ParsedNode):
    def __init__(self, javalang_node):
        super().__init__(javalang_node)
        if not self._decl:
            return
        self.token = f'{self._decl.path}'


class Declaration(ParsedNode):
    def __init__(self, javalang_node):
        super().__init__(javalang_node)
        if not self._decl:
            return
        self.token = f'{self._decl.name}'


class Type(ParsedNode):
    def __init__(self, javalang_node):
        super().__init__(javalang_node)
        if not self._decl:
            return
        self.token = f'{self._decl.name}' \
                     f'{self._decl.dimensions if self._decl.dimensions else ""}'


class Expression(ParsedNode):
    def __init__(self, javalang_node):
        super().__init__(javalang_node)
        self._core = ''
        self._update()

    def _update(self):
        if not self._decl:
            return
        self.token = f'{self._decl.prefix_operators if self._decl.prefix_operators else ""}' \
                     f'{f"{self._decl.qualifier}." if self._decl.qualifier else ""}' \
                     f'{self._core}' \
                     f'{self._decl.postfix_operators if self._decl.postfix_operators else ""}'


class Invocation(Expression):
    def __init__(self, javalang_node):
        super(Invocation, self).__init__(javalang_node)
        self._core = self._decl.member
        self._update()


class Reference(Expression):
    def __init__(self, javalang_node):
        super(Reference, self).__init__(javalang_node)
        self._core = self._decl.member
        self._update()


class Literal(Expression):
    def __init__(self, javalang_node):
        super(Literal, self).__init__(javalang_node)
        self._core = self._decl.value
        self._update()


class Assignment(ParsedNode):
    def __init__(self, javalang_node):
        super().__init__(javalang_node)
        if not self._decl:
            return
        self.token = f'{self._decl.type}'


class BinaryOperation(ParsedNode):
    def __init__(self, javalang_node):
        super().__init__(javalang_node)
        if not self._decl:
            return
        self.token = f'{self._decl.operator}'


class JavaST:
    def __init__(self, source_code):
        self._javalang_root = javalang.parse.parse(source_code)
        self._node_map = defaultdict(lambda: lambda javalang_node: ParsedNode(javalang_node), dict(
            CompilationUnit=lambda javalang_node: CompilationUnit(javalang_node),
            Import=lambda javalang_node: Import(javalang_node),
            ClassDeclaration=lambda javalang_node: Declaration(javalang_node),
            MethodDeclaration=lambda javalang_node: Declaration(javalang_node),
            FormalParameter=lambda javalang_node: Declaration(javalang_node),
            VariableDeclarator=lambda javalang_node: Declaration(javalang_node),
            Literal=lambda javalang_node: Literal(javalang_node),
            MemberReference=lambda javalang_node: Reference(javalang_node),
            MethodInvocation=lambda javalang_node: Invocation(javalang_node),
            Assignment=lambda javalang_node: Assignment(javalang_node),
            BinaryOperation=lambda javalang_node: BinaryOperation(javalang_node),
            Type=lambda javalang_node: Type(javalang_node),
            BasicType=lambda javalang_node: Type(javalang_node),
            ReferenceType=lambda javalang_node: Type(javalang_node),
        ))
        self.root = ParsedNode(None)
        self.nodes = defaultdict(int)
        self.__repr = []
        self.__traverse()

    def __traverse(self):
        parent_stack = []
        prev_depth = 0
        prev_node = ParsedNode(None)
        for path, node in self._javalang_root:
            depth = len(path)
            name = type(node).__name__
            self.nodes[name] += 1
            parsed = self._node_map[name](node)

            if depth == 0:
                self.root = parsed
            elif depth < prev_depth:
                while True:
                    d, n = parent_stack.pop()
                    if d < depth:
                        prev_node = n
                        prev_depth = d
                        parent_stack.append((d, n))
                        _, parent = parent_stack[-1]
                        parent.children.append(parsed)
                        parsed.parent = parent
                        break
            elif depth >= prev_depth:
                if depth > prev_depth:
                    parent_stack.append((prev_depth, prev_node))
                _, parent = parent_stack[-1]
                parent.children.append(parsed)
                parsed.parent = parent

            prev_node = parsed
            prev_depth = depth

    def __traverse_repr(self, root: ParsedNode = None, depth: int = 0):
        if not root:
            self.__repr = []
            root = self.root
        indent = '  ' * depth
        self.__repr.append(f'{indent}{root}')
        for c in root.children:
            if c:
                self.__traverse_repr(c, depth + 1)
        return '\n'.join(self.__repr)

    def __repr__(self):
        return self.__traverse_repr().strip('\n')

    def __traverse_json(self, root: ParsedNode = None):
        n = root if root else self.root
        jo = dict()
        if n.type and n.type != 'None':
            jo['Type'] = n.type
        if n.token and n.token != 'None':
            jo['Token'] = n.token
        if n.parent:
            if (n.parent.type and n.parent.type != 'None') \
                    or (n.parent.token and n.parent.token != 'None'):
                jo['Parent'] = dict()
            if n.parent.type and n.parent.type != 'None':
                jo['Parent']['Type'] = n.parent.type
            if n.parent.token and n.parent.token != 'None':
                jo['Parent']['Token'] = n.parent.token
        if len(n.children) > 0:
            jo['Children'] = []

        for c in n.children:
            jo['Children'].append(self.__traverse_json(c))

        return jo

    def as_json(self):
        return self.__traverse_json()


def build_syntax_tree(args):
    path = args.path
    pre, ext = os.path.splitext(path)
    with open(path, 'r', encoding='utf-8') as fp:
        tree = JavaST(source_code=fp.read())

    with open(pre + '.javast', 'w', encoding='utf-8') as fp:
        print(tree, file=fp, sep='')

    print(f'Generated {pre + ".javast"}')

    if args.json:
        with open(pre + '.javast.json', 'w', encoding='utf-8') as fp:
            json.dump(tree.as_json(), fp, indent=4)

        print(f'Generated {pre + ".javast.json"}')

    tokens = sorted(tree.nodes.keys())
    for i in tokens:
        print(f'{i}: {tree.nodes[i]}')


argparser = argparse.ArgumentParser(description='Build syntax tree for Java source code.')
argparser.add_argument('path', help='path to .java file', action='store')
argparser.add_argument('-j', '--json', help='additionaly stores the tree as a JSON object.', action='store_const', const=True, default=False)
build_syntax_tree(argparser.parse_args())
