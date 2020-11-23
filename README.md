# yuki ❄️
in which I play around with bidirectional typechecking

`pip3 install mypy parsy`, then `mypy yuki.py` and `python3 yuki.py`

it's very WIP, but one day this kind of thing will typecheck:

```ml
(* this is the very cool and good type of integer multiplication in this type system *)
(* & is taking the intersection of function types: *)
mul : (pos ->pos ->pos ) & (pos ->zero->zero) & (pos ->neg ->neg )
    & (zero->pos ->zero) & (zero->zero->zero) & (zero->neg ->zero)
    & (neg ->pos ->neg ) & (neg ->zero->zero) & (neg ->neg ->pos )
mul = __builtin_mul

(* and | is taking the union of value types: *)
square : (zero|pos|neg) -> (zero|pos)
square = fn x => mul x x
```
