# pip install parsy mypy
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from parsy import string, regex, generate, whitespace # type: ignore
from typing import Callable, Mapping, Iterator, List, Set, NoReturn, Union
import inspect

def log(x):
    print(len(inspect.stack()) * '  ' + x)

ws = whitespace.optional()

#####################################################################
#
#  Types
#
#####################################################################

@dataclass(frozen=True)
class YPos:
    def __str__(x): return 'pos'
pPos = string('pos').result(YPos())

@dataclass(frozen=True)
class YZero:
    def __str__(x): return 'zero'
pZero = string('zero').result(YZero())

@dataclass(frozen=True)
class YNeg:
    def __str__(x): return 'neg'
pNeg = string('neg').result(YNeg())

@dataclass(frozen=True)
class YBool:
    def __str__(x): return 'bool'
pBool = string('bool').result(YBool())

@dataclass(frozen=True)
class YList:
    member: 'YType'
    def __str__(x): return f'list[{x.member}]'
@generate("pList")
def pList():
    t = yield string('list') >> ws >> string('[') >> ws >> pType << ws << string(']')
    return YList(t)

@dataclass(frozen=True)
class YArr:
    source: 'YType'
    target: 'YType'
    def __str__(x): return f'({x.source} -> {x.target})'
@generate("pArr")
def pArr():
    source = yield pType0 << ws << string('->') << ws
    target = yield pType
    return YArr(source, target)

@dataclass(frozen=True)
class YUnion:
    left: 'YType'
    right: 'YType'
    def __str__(x): return f'({x.left} | {x.right})'
@generate("pUnion")
def pUnion():
    source = yield pType0 << ws << string('|') << ws
    target = yield pType
    return YUnion(source, target)

@dataclass(frozen=True)
class YIntersection:
    left: 'YType'
    right: 'YType'
    def __str__(x): return f'({x.left} & {x.right})'
@generate("pIntersection")
def pIntersection():
    source = yield pType0 << ws << string('&') << ws
    target = yield pType
    return YIntersection(source, target)

@generate("pParenType")
def pParenType():
    return (yield string('(') >> ws >> pType << ws << string(')'))

YType = Union[YPos, YZero, YNeg, YBool, YList, YArr, YUnion, YIntersection]
pType0 = pPos | pZero | pNeg | pBool | pList | pParenType
pType = pArr | pUnion | pIntersection | pType0

'''
is (A->B)&(X->Y) a subtype of (A|X)->(B|Y)?
yeah I think so. let's... let's special case this:
'''

def eliminate_intersection(t: YIntersection):
    nodes: List[YType] = [t]
    sources: Set[YType] = set()
    targets: Set[YType] = set()
    while nodes:
        u = nodes.pop(0)
        if isinstance(u, YIntersection):
            nodes.append(u.left)
            nodes.append(u.right)
        elif isinstance(u, YArr):
            sources.add(u.source)
            targets.add(u.target)
        else:
            raise
    return YArr(reduce(YUnion, sources), reduce(YUnion, targets))

def is_subtype(t1, t2):
    # log(f'--- Checking if {t1} <: {t2}')
    if t1 == t2: return True
    if isinstance(t1, YUnion):
        return is_subtype(t1.left, t2) and is_subtype(t1.right, t2)
    elif isinstance(t2, YUnion):
        return is_subtype(t1, t2.left) or is_subtype(t1, t2.right)
    elif isinstance(t1, YIntersection):
        return is_subtype(eliminate_intersection(t1), t2)
    elif isinstance(t2, YIntersection):
        return is_subtype(t1, t2.left) and is_subtype(t1, t2.right)
    elif isinstance(t1, YArr) and isinstance(t2, YArr):
        return is_subtype(t2.source, t1.source) and is_subtype(t1.target, t2.target)
    return False

def max_type(t1, t2):
    if is_subtype(t1, t2):
        return t2
    elif is_subtype(t2, t1):
        return t1
    else:
        return YUnion(t1, t2)

#####################################################################
#
#  Expressions
#
#####################################################################

@dataclass(frozen=True)
class YLitNumber:
    value: int
    def __str__(x): return str(x.value)
pLitNumber = regex(r'-?[0-9]+').map(int).map(YLitNumber)

@dataclass(frozen=True)
class YLitBool:
    value: bool
    def __str__(x): return str(x.value).lower()
pTrue = string('true').result(True).map(YLitBool)
pFalse = string('false').result(False).map(YLitBool)
pLitBool = pTrue | pFalse

@dataclass(frozen=True)
class YVar:
    name: str
    def __str__(x): return f'`{x.name}'
pVar = regex(r'\b[a-zA-Z]+\b').map(YVar)

@dataclass(frozen=True)
class YLam:  # fn x => e
    variable: YVar
    body: 'YExpr'
    def __str__(x): return f'(fn {x.variable.name} => {x.body})'
@generate
def pLam():
    variable = yield string('fn') >> ws >> pVar << ws << string('=>') << ws
    body = yield pExpr
    return YLam(variable, body)

@dataclass(frozen=True)
class YBuiltin:
    type: YType
    impl: Callable
    name: str
    def __str__(x): return x.name

@dataclass(frozen=True)
class YAnno:  # (e : t)
    expression: 'YExpr'
    annotatedType: YType
    def __str__(x): return f'({x.expression} : {x.annotatedType})'
@generate
def pAnno():
    expression = yield string('(') >> ws >> pExpr << ws << string(':') << ws
    annotatedType = yield pType << ws << string(')')
    return YAnno(expression, annotatedType)

@dataclass(frozen=True)
class YApp:  # e1 e2
    function: 'YExpr'
    argument: 'YExpr'
    def __str__(x): return f'({x.function}) {x.argument}'
@generate
def pApp():
    function = yield pLow << ws
    arguments = yield pLow.sep_by(whitespace, min=1)
    return reduce(YApp, arguments, function)

@dataclass(frozen=True)
class YIfSubtype:  # if var <: t then e1 else e2 end
    scrutinee: 'YVar'
    type: 'YType'
    thenBody: 'YExpr'
    elseBody: 'YExpr'
    def __str__(x): return f'if {x.scrutinee} <: {x.type} then {x.thenBody} else {x.elseBody} end'
@generate
def pIfSubtype():
    scrutinee = yield string('if') >> ws >> pVar
    type = yield ws >> string('<:') >> ws >> pType
    thenBody = yield ws >> string('then') >> ws >> pLow
    elseBody = yield ws >> string('else') >> ws >> pLow << ws << string('end')
    return YIfSubtype(scrutinee, type, thenBody, elseBody)
YExpr = Union[YLitNumber, YLitBool, YBuiltin, YVar, YLam, YAnno, YApp, YIfSubtype]

@generate
def pParen():
    return (yield string('(') >> ws >> pExpr << ws << string(')'))
pLow = pLitBool | pLitNumber | pIfSubtype | pVar | pAnno | pParen
pExpr = pLam | pApp | pLow

#####################################################################
#
#  Type checking
#
#####################################################################

Context = Mapping[str, YType]

def dictstr(d):
    return '{' + ', '.join(f'{k}: {v}' for k, v in d.items()) + '}'

def intersection_members(t: YType) -> Iterator[YType]:
    if isinstance(t, YIntersection):
        yield from intersection_members(t.left)
        yield from intersection_members(t.right)
    else:
        yield t

class IllTyped(Exception): pass
class UnknownVariable(Exception): pass
class Limitation(Exception): pass
class NeedAnnotation(Exception): pass

def synth(Γ: Context, expr: YExpr) -> YType:
    log(f'Synthing type for {expr}.') # {dictstr(Γ)}')
    if isinstance(expr, YLitNumber):
        return YPos() if expr.value > 0 else YNeg() if expr.value < 0 else YZero()
    elif isinstance(expr, YLitBool):
        return YBool()
    elif isinstance(expr, YBuiltin):
        return expr.type
    elif isinstance(expr, YVar):
        if expr.name not in Γ:
            raise UnknownVariable(f"I don't know what '{expr.name}' refers to.")
        return Γ[expr.name]
    elif isinstance(expr, YApp):
        tf = synth(Γ, expr.function)
        
        if isinstance(tf, YArr):
            check(Γ, expr.argument, tf.source)
            return tf.target
        elif isinstance(tf, YIntersection):
            ta = synth(Γ, expr.argument)
            targets: Set[YType] = set()
            for i in intersection_members(tf):
                if not isinstance(i, YArr):
                    raise Limitation(f"I don't understand intersections of non-function types.")
                log(f'~~~ Now checking: {i.source} <: {ta}')
                # I am almost 100% sure this is wrong
                if is_subtype(i.source, ta):
                    targets.add(i.target)
            if targets:
                # I am more than 100% sure this is wrong...
                # TODO     have thoughts. brain full
                if all(isinstance(x, YArr) for x in targets):
                    return reduce(YIntersection, targets)
                return reduce(YUnion, targets)
            else:
                raise IllTyped(f"blahh\n{expr.function} : {tf}\n{expr.argument} : {ta}\n{targets}")
        else:
            raise IllTyped(f"You can't apply {expr.function} : {tf} to anything, because it's not a function.")
    elif isinstance(expr, YIfSubtype):
        log(f'!!! Refining type of {expr.scrutinee} to {expr.type} in {expr.thenBody}...')
        tt = synth({**Γ, expr.scrutinee.name: expr.type}, expr.thenBody)
        tf = synth(Γ, expr.elseBody)
        log(f'max_type({tt}, {tf}) == {max_type(tt, tf)}')
        return max_type(tt, tf)
    elif isinstance(expr, YAnno):
        check(Γ, expr.expression, expr.annotatedType)
        return expr.annotatedType
    else:
        raise Limitation(f"I don't know how to synthesize a type for {expr}. You'll need to annotate the type for me.")

def check(Γ: Context, expr: YExpr, t: YType) -> None:
    log(f'Checking that {expr} is a {t}.')
    #
    #   this stuff is commented out because it's synth()'s job
    #
    #if isinstance(expr, YLitNumber):
    #    if t != YInt():
    #        raise ValueError(f"{expr} is a number, but I was expecting a {t}.")
    #elif isinstance(expr, YLitBool):
    #    if t != YBool():
    #        raise ValueError(f"{expr} is a bool, but I was expecting a {t}.")
    #elif isinstance(expr, YBuiltin):
    #    if t != expr.type:
    #        raise ValueError(f"{expr} is a {expr.type}, but I was expecting a {t}.")
    #elif isinstance(expr, YVar):
    #    if expr.name not in Γ:
    #        raise ValueError(f"I expected a {t} here, but I found '{expr.name}', which I don't know what it refers to.")
    #    if t != Γ[expr.name]:
    #        raise ValueError(f"I expected a {t} here, but I found '{expr.name}' of type {t}.")
    if isinstance(expr, YLam):
        if not isinstance(t, YArr):
            raise IllTyped(f"{expr} is a lambda, but I was expecting a {t}.")
        check({**Γ, expr.variable.name: t.source}, expr.body, t.target)
    else:
        # turn around
        t0 = synth(Γ, expr)
        if not is_subtype(t0, t):
            raise IllTyped(f"I synthesized {t0} for {expr}, but it's not a subtype of {t}.")

#####################################################################
#
#  Evaluation
#
#####################################################################

Environment = Mapping[str, YExpr]

def evaluate(E: Environment, expr: YExpr) -> YExpr:
    if isinstance(expr, YLitNumber):
        return expr
    elif isinstance(expr, YLitBool):
        return expr
    elif isinstance(expr, YLam):
        return expr
    elif isinstance(expr, YBuiltin):
        return expr
    elif isinstance(expr, YAnno):
        return expr.expression
    elif isinstance(expr, YVar):
        return E[expr.name]
    elif isinstance(expr, YApp):
        f = evaluate(E, expr.function)
        if isinstance(f, YBuiltin):
            return f.impl(evaluate(E, expr.argument))
        elif isinstance(f, YLam):
            E_ = {**E, f.variable.name: evaluate(E, expr.argument)}
            return evaluate(E_, f.body)
        else:
            raise IllTyped(f"I can't apply {expr.function}, because it's not a function. If you see this, then the type-checker isn't doing it's job!")
    elif isinstance(expr, YIfSubtype):
        if expr.scrutinee.name not in E:
            raise UnknownVariable(f"You used '{expr.scrutinee.name}' in a subtype check, but I don't know what that refers to.")
        if is_subtype(synth({}, E[expr.scrutinee.name]), expr.type):
            return evaluate(E, expr.thenBody)
        else:
            return evaluate(E, expr.elseBody)
    else:
        raise Limitation(f"I don't know how to evaluate {expr}.")

#####################################################################
#
#  Prelude
#
#####################################################################

def closure(lam):
    # The "type" here actually doesn't matter: it will only be unveiled during execution,
    # so the typechecker never sees it.
    return YBuiltin(type=YZero(), name='(built-in closure)', impl=lam)

prelude = {
    'sqrt': YBuiltin(
        type=pType.parse('(pos->pos) & (zero->zero)'),
        impl=lambda x: YLitNumber(int(x.value**0.5)),
        name='sqrt',
    ),
    'negate': YBuiltin(
        type=pType.parse('(neg->pos) & (zero->zero) & (pos->neg)'),
        impl=lambda x: YLitNumber(-x.value),
        name='negate',
    ),
    'add': YBuiltin(
        type=pType.parse('(pos->pos->pos) & (pos->zero->pos) & (pos->neg->(pos|zero|neg)) & (zero->pos->pos) & (zero->zero->zero) & (zero->neg->neg) & (neg->pos->(pos|zero|neg)) & (neg->zero->neg) & (neg->neg->neg)'),
        impl=lambda x: closure(lambda y: YLitNumber(x.value + y.value)),
        name='add',
    ),
}

ast = pExpr.parse('((fn x => if x <: (pos|zero) then (sqrt x) else -1 end) : (pos|zero|neg) -> (pos|zero|neg)) 16')
# ast = pExpr.parse('((fn x => add 3 x) : pos -> pos) 16')
# ast = pExpr.parse('add 3')
print(ast)
Γprelude = {k: v.type for k, v in prelude.items()}
print(synth(Γprelude, ast))
print(evaluate(prelude, ast))