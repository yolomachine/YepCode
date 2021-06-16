from __future__ import annotations

import json
import os
import javalang
import argparse
from typing import Any, Optional, Dict, List
from collections import defaultdict


class ParsedNode:
    def __init__(self, **kwargs):
        self._decl = kwargs.get('jln', None)
        self.type = kwargs.get('type', self.__try_override_type())
        self.token = kwargs.get('token', None)
        self.parent = kwargs.get('parent', None)
        self.children = kwargs.get('children', [])

    def __try_override_type(self) -> str:
        t = type(self._decl).__name__ if self._decl else f''
        return self.__type_map[t] or t

    @property
    def __type_map(self) -> defaultdict[Any, str]:
        return defaultdict(str, dict(
            VariableDeclarator='Variable',
        ))

    def __repr__(self):
        return f'{self.type}' + (f'`{self.token}`' if self.token else '')

    @staticmethod
    def clone(node: ParsedNode) -> ParsedNode:
        clone = ParsedNode(type=node.type, token=node.token)
        for c in node.children:
            child_clone = ParsedNode.clone(c)
            child_clone.relink(clone)
        return clone

    def unlink(self) -> None:
        if self.parent:
            index = self.parent.children.index(self)
            self.parent.children = self.parent.children[:index] + self.parent.children[index+1:]
            self.parent = None

    def relink(self, node) -> None:
        if node:
            self.unlink()
            self.parent = node
            node.children.append(self)

    def child_of_type(self, node_type) -> Optional[ParsedNode]:
        for c in self.children:
            if c.type == node_type:
                return c
        return None


class CustomNode(ParsedNode):
    def __init__(self):
        super().__init__(type=self.__class__.__name__)


class CustomTokenizedNode(CustomNode):
    def __init__(self, name):
        super(CustomTokenizedNode, self).__init__()
        self.children.append(ParsedNode(token=name, parent=self))


class Package(CustomTokenizedNode):
    def __init__(self, name):
        super(Package, self).__init__(name)


class PackageMember(CustomTokenizedNode):
    def __init__(self, path):
        super(PackageMember, self).__init__(path)


class Identifier(CustomTokenizedNode):
    def __init__(self, name):
        super(Identifier, self).__init__(name)


class Value(CustomTokenizedNode):
    def __init__(self, name):
        super(Value, self).__init__(name)


class Accessor(CustomTokenizedNode):
    def __init__(self, name):
        super(Accessor, self).__init__(name)


class OperatorSpecification(CustomTokenizedNode):
    def __init__(self, name):
        super(OperatorSpecification, self).__init__(name)


class PrefixOperators(CustomNode):
    def __init__(self, operators):
        super(PrefixOperators, self).__init__()
        self.children = operators
        for c in self.children:
            c.parent = self


class PostfixOperators(CustomNode):
    def __init__(self, operators):
        super(PostfixOperators, self).__init__()
        self.children = operators
        for c in self.children:
            c.parent = self


class Signature(CustomNode):
    def __init__(self, nodes=None):
        super(Signature, self).__init__()
        self.children = nodes or []
        for c in self.children:
            c.parent = self


class Body(CustomNode):
    def __init__(self, nodes=None):
        super(Body, self).__init__()
        self.children = nodes or []
        for c in self.children:
            c.parent = self


class Condition(CustomNode):
    def __init__(self, nodes=None):
        super(Condition, self).__init__()
        self.children = nodes or []
        for c in self.children:
            c.parent = self


class InvocationArguments(CustomNode):
    def __init__(self, nodes=None):
        super(InvocationArguments, self).__init__()
        self.children = nodes or []
        for c in self.children:
            c.parent = self


class CompilationUnit(ParsedNode):
    def __init__(self, javalang_node):
        super().__init__(jln=javalang_node)
        if not self._decl:
            return
        package = self._decl.package
        if package:
            self.children.append(Package(package))
        for c in self.children:
            c.parent = self


class Import(ParsedNode):
    def __init__(self, javalang_node):
        super().__init__(jln=javalang_node)
        if not self._decl:
            return
        package = self._decl.path
        if package:
            self.children.append(PackageMember(package))
        for c in self.children:
            c.parent = self


class Declaration(ParsedNode):
    def __init__(self, javalang_node):
        super().__init__(jln=javalang_node)
        if not self._decl:
            return
        identifier = self._decl.name
        if identifier:
            self.children.append(Identifier(identifier))
        for c in self.children:
            c.parent = self


class Type(ParsedNode):
    def __init__(self, javalang_node):
        super().__init__(jln=javalang_node)
        if not self._decl:
            return
        decl_type = self._decl.name + ('[]' * len(self._decl.dimensions) if self._decl.dimensions else '')
        if decl_type:
            self.children.append(Identifier(decl_type))
        for c in self.children:
            c.parent = self


class Expression(ParsedNode):
    def __init__(self, javalang_node):
        super().__init__(jln=javalang_node)
        self._core = ParsedNode()

    def _build(self) -> None:
        if not self._decl:
            return

        if self._decl.qualifier:
            self.children.append(Accessor(self._decl.qualifier))

        self.children.append(self._core)

        if self._decl.prefix_operators:
            operators = self._decl.prefix_operators
            if operators:
                self.children.append(PrefixOperators([UnaryOperation(i) for i in operators]))

        if self._decl.postfix_operators:
            operators = self._decl.postfix_operators
            if operators:
                self.children.append(PostfixOperators([UnaryOperation(i) for i in operators]))

        for c in self.children:
            c.parent = self

    @property
    def __map(self) -> defaultdict[Any, str]:
        return defaultdict(lambda: 'Custom', {
            '+': 'Plus',
            '-': 'Minus',
            '++': 'Increment',
            '--': 'Decrement',
            '!': 'LogicalComplement'
        })


class Invocation(Expression):
    def __init__(self, javalang_node):
        super(Invocation, self).__init__(javalang_node)
        self._core = Identifier(self._decl.member)
        self._build()


class Reference(Expression):
    def __init__(self, javalang_node):
        super(Reference, self).__init__(javalang_node)
        self._core = Identifier(self._decl.member)
        self._build()


class Literal(Expression):
    def __init__(self, javalang_node):
        super(Literal, self).__init__(javalang_node)
        self._core = Value(self._decl.value)
        self._build()


class Assignment(ParsedNode):
    def __init__(self, javalang_node):
        super().__init__(jln=javalang_node)
        if not self._decl:
            return
        self.children.append(OperatorSpecification(self.__map[self._decl.type]))
        for c in self.children:
            c.parent = self

    @property
    def __map(self) -> defaultdict[Any, str]:
        return defaultdict(lambda: 'Regular', {
            '+=': 'Addition',
            '-=': 'Subtraction',
            '*=': 'Multiplication',
            '/=': 'Division',
            '%=': 'Modulo',
            '&=': 'BitwiseAnd',
            '|=': 'BitwiseOr',
            '^=': 'ExclusiveOr',
            '>>=': 'ShiftRight',
            '<<=': 'ShiftLeft',
        })


class UnaryOperation(CustomNode):
    def __init__(self, operator):
        super().__init__()
        self.children.append(OperatorSpecification(self.__map[operator]))
        for c in self.children:
            c.parent = self

    @property
    def __map(self) -> defaultdict[Any, str]:
        return defaultdict(lambda: 'Custom', {
            '+': 'Plus',
            '-': 'Minus',
            '++': 'Increment',
            '--': 'Decrement',
            '!': 'LogicalComplement',
        })


class BinaryOperation(ParsedNode):
    def __init__(self, javalang_node):
        super().__init__(jln=javalang_node)
        if not self._decl:
            return
        self.children.append(OperatorSpecification(self.__map[self._decl.operator]))
        for c in self.children:
            c.parent = self

    @property
    def __map(self) -> defaultdict[Any, str]:
        return defaultdict(lambda: 'Custom', {
            '==': 'Equals',
            '!=': 'NotEquals',
            '>': 'GreaterThan',
            '<': 'LessThan',
            '>=': 'GreaterThanOrEqual',
            '<=': 'LessThanOrEqual',
            '&&': 'LogicalAnd',
            '||': 'LogicalOr',
            '+': 'Addition',
            '-': 'Subtraction',
            '*': 'Multiplication',
            '/': 'Division',
            '%': 'Modulo',
            '&': 'BitwiseAnd',
            '|': 'BitwiseOr',
            '^': 'ExclusiveOr',
            '>>': 'ShiftRight',
            '<<': 'ShiftLeft',
        })


class JavaST:
    def __init__(self, source_code: str = None, root: ParsedNode = None):
        self.__javalang_root = javalang.parse.parse(source_code) if source_code else None
        self.__node_map = defaultdict(lambda: lambda javalang_node: ParsedNode(jln=javalang_node), dict(
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
        self.root = root or ParsedNode()
        self.__repr = []
        if source_code:
            self.__traverse()
            self.__fixup()

    def __traverse(self) -> None:
        parent_stack = []
        prev_depth = 0
        prev_node = ParsedNode()
        for path, node in self.__javalang_root:
            depth = len(path)
            name = type(node).__name__
            parsed = self.__node_map[name](node)

            if depth == 0:
                self.root = parsed
            elif depth < prev_depth:
                while True:
                    d, n = parent_stack.pop()
                    if d < depth:
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

    def __fixup(self, root: ParsedNode = None) -> None:
        root = root or self.root
        children = root.children[:]
        signature, body, condition = None, None, None
        for c in children:
            if root.type in [
                'ClassDeclaration',
                'MethodDeclaration',
            ]:
                if root.type == 'MethodDeclaration':
                    body = root.child_of_type('Body')
                    if not body:
                        body = Body()
                        body.parent = root
                        root.children = [body] + root.children

                signature = root.child_of_type('Signature')
                if not signature:
                    signature = Signature()
                    signature.parent = root
                    root.children = [signature] + root.children

                if c.type in [
                    'Identifier',
                    'BasicType',
                    'ReferenceType',
                    'FormalParameter',
                ]:
                    c.relink(signature)
                elif body:
                    c.relink(body)

            if 'Type' in root.type:
                if c.type not in [
                    'Identifier',
                ]:
                    if root.parent.type == 'Signature' and c.type != 'FormalParameter':
                        body = root.parent.parent.child_of_type('Body')
                        c.relink(body)
                    elif root.parent.type == 'ClassCreator':
                        if c._decl in root.parent._decl.arguments:
                            arguments = root.parent.child_of_type('InvocationArguments')
                            if not arguments:
                                arguments = InvocationArguments()
                                arguments.parent = root.parent
                                root.parent.children = root.parent.children + [arguments]
                            c.relink(arguments)
                    else:
                        c.relink(root.parent)

            if root.type == 'BinaryOperation':
                if children.index(c) >= 3:
                    c.relink(root.parent)

            if root.type in [
                'WhileStatement',
                'IfStatement',
            ]:
                if c.type not in [
                    'BlockStatement',
                ]:
                    condition = root.child_of_type('Condition')
                    if not condition:
                        condition = Condition()
                        condition.parent = root
                        root.children = [condition] + root.children
                    c.relink(condition)

            if root.type in [
                'MethodInvocation',
                'ClassCreator',
            ]:
                if c.type not in [
                    'Accessor',
                    'Identifier',
                    'PrefixOperators',
                    'PostfixOperators',
                    'ReferenceType',
                    'BasicType',
                ]:
                    if c._decl in root._decl.arguments:
                        arguments = root.child_of_type('InvocationArguments')
                        if not arguments:
                            arguments = InvocationArguments()
                            arguments.parent = root
                            root.children = root.children + [arguments]
                        c.relink(arguments)

            self.__fixup(c)

    def __traverse_repr(self, root: ParsedNode = None, depth: int = 0) -> str:
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

    def __traverse_json(self, root: ParsedNode = None) -> Dict[str, Any]:
        root = root or self.root
        jo = dict()
        if root.type and root.type != 'None':
            jo['Type'] = root.type
        if root.token and root.token != 'None':
            jo['Token'] = root.token
        if len(root.children) > 0:
            jo['Children'] = []

        for c in root.children:
            jo['Children'].append(self.__traverse_json(c))

        return jo

    def __traverse_statements(self, root: ParsedNode = None, sequence: List[JavaST] = None) -> List[JavaST]:
        root = root or ParsedNode.clone(self.root)
        sequence = sequence or []
        for c in root.children[:]:
            if c.type in [
                'ClassDeclaration',
                'ConstructorDeclaration',
                'MethodDeclaration',
                'FieldDeclaration',
                'LocalVariableDeclaration',
                'StatementExpression',
                'ForStatement',
                'WhileStatement',
                'IfStatement',
                'ReturnStatement',
                'TryStatement',
                'CatchClause',
                'MethodInvocation',
            ]:
                c.unlink()
                sequence.append(JavaST(root=c))
            self.__traverse_statements(c, sequence)
        return sequence

    def as_json(self) -> Dict[str, Any]:
        return self.__traverse_json()

    def as_vocab(self) -> Dict[str, int]:
        vocab = defaultdict(int)
        queue = [self.root]
        while len(queue) > 0:
            node = queue[0]
            queue = queue[1:]
            for c in node.children:
                queue.append(c)
            vocab[node.type or node.token] += 1
        return vocab

    def as_statement_sequence(self) -> List[JavaST]:
        return self.__traverse_statements()


def build_syntax_tree(args) -> None:
    path = args.path
    pre, ext = os.path.splitext(path)
    try:
        with open(path, 'r', encoding='utf-8') as fp:
            tree = JavaST(source_code=fp.read())
        print(f'{path} read successfully')
    except:
        print(f'Couldn\'t read {path}')

    try:
        with open(pre + '.javast', 'w', encoding='utf-8') as fp:
            print(tree, file=fp, sep='')
        print(f'Generated {pre + ".javast"}')
    except:
        print(f'Couldn\'t generate {pre + ".javast"}')

    if args.json:
        try:
            with open(pre + '.javast.json', 'w', encoding='utf-8') as fp:
                json.dump(tree.as_json(), fp, indent=4)
            print(f'Generated {pre + ".javast.json"}')
        except:
            print(f'Couldn\'t generate {pre + ".javast.json"}')

    try:
        vocab = tree.as_vocab()
        with open(pre + '.javast.vocab', 'w', encoding='utf-8') as fp:
            for key in vocab:
                print(f'{key}: {vocab[key]}', file=fp)
            print(f'\nUnique: {len(vocab.keys())}', file=fp)
            print(f'Total: {sum(vocab.values())}', file=fp)
        print(f'Generated {pre + ".javast.vocab"}')
    except:
        print(f'Couldn\'t generate {pre + ".javast.vocab"}')

    try:
        seq = tree.as_statement_sequence()
        with open(pre + '.javast.seq', 'w', encoding='utf-8') as fp:
            for block in seq:
                print(f'{block}\n', file=fp)
        print(f'Generated {pre + ".javast.seq"}')
    except:
        print(f'Couldn\'t generate {pre + ".javast.seq"}')


argparser = argparse.ArgumentParser(description='Build syntax tree for Java source code.')
argparser.add_argument('path', help='path to .java file', action='store')
argparser.add_argument('-j', '--json', help='additionaly stores the tree as a JSON object.', action='store_const', const=True, default=False)
build_syntax_tree(argparser.parse_args())
