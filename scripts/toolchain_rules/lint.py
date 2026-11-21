"""Scan Bend sources for patterns the official checker rejects.

Run from the repository root before a checker run:
  python -B scripts/toolchain_rules/lint.py FILE_OR_DIR [...]
It only reads the files and takes seconds. Each finding names its rule in
docs/toolchain_rules.md. It looks at spelling and match structure only: it
finds neither usage errors nor type mismatches.
"""
import re
import sys
from pathlib import Path

DEF = re.compile(r"^(?:@unsafe\s+)?def\s+([\w.]+)\??\(", re.M)
LAW = re.compile(r"^law\s+([\w.]+):\n((?:  for .*\n)+)", re.M)
IMPORT = re.compile(r"^import\s+(\S+\.bend)\s+as\s+(\w+)\s*$", re.M)
CALL = re.compile(r"(?<![\w.])((?:[A-Z]\w*\.)?[a-z_]\w*)\(")
BIG_WORD = re.compile(r"U32\.to_nat\(\s*(\d{6,})\s*\)")
CLOSED_LIMIT = re.compile(r"U32\.to_nat\(\s*[\w.]+\(\s*\w+\.limits\(\)\s*\)\s*\)")
BIG_NAT = re.compile(r"(?<![\w.])(\d{6,})n\b")
MATCH = re.compile(r"^\s*match\s+(.*):\s*$")
LET = re.compile(r"^(\s*)\+?[a-z_]\w*(?:\s+\+?[a-z_]\w*)*\s*(?<![=!<>])=(?!=)\s*(.*)$")
DESTRUCTURE = re.compile(r"^(\s*)(\([^=]*\)|[A-Z][\w.]*\{[^=]*\})\s*(?<![=!<>])=(?!=)\s*(.+)$")
NAME = re.compile(r"^[a-z_]\w*$")
BARE_CTOR_LET = re.compile(r"^\s*\+?[a-z_]\w*\s*(?<![=!<>])=(?!=)\s*[A-Z][\w.]*\{")
LIST_PATTERN = re.compile(r"^\s*case\s+.*[\w}]\s+\[\]")


# Each rule: (what was found, why the checker rejects it, what to do).
RULES = {
    "A1": ("a binder matched after a later one",
           "The checker matches parameters and the fields they bind from left to right. Once a binder is "
           "destructed, every binder before it can no longer be matched (can't be matched in this position).",
           "Match binders in parameter order: put the parameter matched first earlier in the parameter list, "
           "or match the earlier binder in the same case line."),
    "A2": ("a match or destructuring after a let",
           "The checker compiles a body into a case tree. At the first let, every parameter not yet "
           "destructed becomes a lambda and can no longer be matched (can't be matched in this position).",
           "Move every match and destructuring of parameters before the first let. To branch on a computed "
           "value, put the rest of the body in another def that takes the value as a parameter."),
    "A3": ("a match or destructuring of a computed value",
           "A match applies to a parameter or a bound field, not to a call result or another expression "
           "(a match cannot scrutinize a computed value).",
           "Write a helper def that takes the value as a parameter and matches it there."),
    "A7": ("a pattern followed by []",
           "`name []` parses as an array read, not as two patterns (expected : a term / observed : ']').",
           "Write list patterns as Con{head,tail} and Nil{}."),
    "C1": ("a ~ parameter that is not leading",
           "Template parameters come first in a def's parameter list (only leading binders take ~).",
           "Move every ~ parameter to the front and reorder the arguments at the call sites."),
    "C2": ("a call with excess or non-leading ~ arguments",
           "Only parameters the callee declares with ~ take ~. An extra ~ fails in the parser "
           "(expected : a term / observed : '~'), often reported on a nearby line.",
           "Write ~ on a leading prefix of the callee's ~ parameters, or on none."),
    "C5": ("a use of a def written later in the file",
           "Defs are checked in file order, and live code cannot refer to a def that has no body yet (a def whose body "
           "is implemented before the def is used). There is no mutual recursion and no forward reference.",
           "Write the used def first. For a recursion through a helper, compute the helper's result in a def written "
           "before the recursive one and pass it as a parameter."),
    "C4": ("a self-call that does not decrease",
           "The arguments of a self-call are compared with the columns as matched so far, from left to right. "
           "Each must be passed unchanged until one is smaller: a variable bound inside its column's pattern, or a "
           "constructor rebuilt from such parts (a call smaller than ...). ~ and - parameters are skipped, and "
           "self-calls in types are not checked.",
           "Put the parameter the recursion descends on before every parameter that changes freely, or "
           "recurse in a helper def whose first non-erased parameter is the decreasing one."),
    "D4": ("a let of a constructor without a type annotation",
           "A constructor needs a target type to be checked; alone on the right of a let it cannot be inferred.",
           "Write +x = {Ctor{...} : Type}."),
    "E1": ("a large Nat in a type or expression",
           "Nat is unary. When two types containing a closed large Nat in different spellings are compared, "
           "the checker may unfold it. Structural comparison depth grows with the value and can overflow the stack "
           "(the machine stack overflowed), with no location. Computing with a large Nat literal takes time "
           "proportional to its value even when it does not overflow.",
           "Keep the quantity abstract: make it a parameter (for example +limit: U32) and use it only through "
           "equations and inequalities. Compare concrete numbers on U32 (U32.is_le)."),
    "E1?": ("a default machine limit converted to Nat (a risk, not always a failure)",
            "U32.to_nat(Machine.capacity(Default.limits())) is a closed large Nat that can be computed. "
            "Comparing differently spelled types can expose this computation and overflow the stack, with no location.",
            "Follow the convolution proofs: take +limits: Machine.Limits, write Machine.capacity(limits) in types, "
            "and substitute Default.limits() only at the top public entry. See count_within in "
            "lib/proofs/bounded_u32_arithmetic.bend and lib/proofs/checked_word_completion.bend."),
}


def split_top(text):
    """Split an argument list at top-level commas."""
    parts, depth, current = [], 0, ""
    for char in text:
        if char in "([{<" and not (char == "<" and current.endswith(" ")):
            depth += 1
        elif char in ")]}>" and depth > 0 and not (char == ">" and current.endswith(("=", "-", " "))):
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts


def close_paren(text, start):
    """Index just past the parenthesis that closes the one at start - 1."""
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


TEMPLATES = {}


def templates(path):
    """Leading ~ parameter count of every def in a file."""
    path = path.resolve()
    if path not in TEMPLATES:
        found = {}
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        for match in DEF.finditer(text):
            end = close_paren(text, match.end())
            if end < 0:
                continue
            params = split_top(text[match.end():end])
            leading = 0
            for param in params:
                if not param.startswith("~"):
                    break
                leading += 1
            late = [param for param in params[leading:] if param.startswith("~")]
            found[match.group(1)] = (leading, bool(late), text.count("\n", 0, match.start()) + 1)
        # A law declares its ~ parameters as leading `for ~name: T` lines.
        for match in LAW.finditer(text):
            binders = [line.strip() for line in match.group(2).splitlines()]
            leading = 0
            for binder in binders:
                if not binder.startswith("for ~"):
                    break
                leading += 1
            late = [binder for binder in binders[leading:] if binder.startswith("for ~")]
            found[match.group(1)] = (leading, bool(late), text.count("\n", 0, match.start()) + 1)
        TEMPLATES[path] = found
    return TEMPLATES[path]


def param_names(params):
    """Names and markers of a def's parameters, in order."""
    found = []
    for param in params:
        head = param.split(":", 1)[0].strip()
        marker = head[0] if head[:1] in ("+", "-", "~") else ""
        found.append((head.lstrip("+-~").strip(), marker))
    return found


def pattern_columns(text):
    """Split the patterns of a case line into one entry per scrutinee."""
    tokens, depth, current = [], 0, ""
    for char in text:
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
        if char == " " and depth == 0:
            if current:
                tokens.append(current)
            current = ""
        else:
            current += char
    if current:
        tokens.append(current)
    columns, index = [], 0
    while index < len(tokens):
        token = tokens[index]
        while token.endswith("+") and index + 1 < len(tokens):
            index += 1
            token += " " + tokens[index]
        while index + 2 < len(tokens) and tokens[index + 1] == "<>":
            token += " <> " + tokens[index + 2]
            index += 2
        columns.append(token)
        index += 1
    return columns


def top_split(text, separator):
    """Split text at the first separator outside brackets, or return None."""
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith(separator, index):
            return text[:index], text[index + len(separator):]
    return None


OPAQUE = ("opaque",)
IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")


def term(text, env=None):
    """The constructor shape of a pattern or argument: a variable, a constructor with fields, or opaque."""
    env = env or {}
    text = text.strip()
    while text[:1] in ("+", "-") and text[1:2] != ">":
        text = text[1:].strip()
    if text.startswith("{") and text.endswith("}"):
        annotated = top_split(text[1:-1], " : ")
        if annotated and "==" not in annotated[0]:
            return term(annotated[0], env)
        return OPAQUE
    if text.startswith("(") and text.endswith(")") and close_paren(text, 1) == len(text) - 1:
        return term(text[1:-1], env)
    if IDENTIFIER.match(text):
        return env.get(text, ("var", text))
    cons = top_split(text, "<>")
    if cons:
        return ("ctr", "Con", [term(cons[0], env), term(cons[1], env)])
    number = re.match(r"^(\d+)n(?:\+(.+))?$", text, re.S)
    if number:
        found = term(number.group(2), env) if number.group(2) else ("ctr", "Zero", [])
        for _ in range(int(number.group(1))):
            found = ("ctr", "Succ", [found])
        return found
    if text.startswith("[") and text.endswith("]"):
        found = ("ctr", "Nil", [])
        for item in reversed(split_top(text[1:-1])):
            found = ("ctr", "Con", [term(item, env), found])
        return found
    built = re.match(r"^([A-Z][\w.]*)\{(.*)\}$", text, re.S)
    if built:
        return ("ctr", built.group(1).split(".")[-1], [term(field, env) for field in split_top(built.group(2))])
    return OPAQUE


def binders(shape):
    """The variables of a pattern shape, left to right."""
    if shape[0] == "var":
        return [shape[1]]
    if shape[0] == "ctr":
        return [name for field in shape[2] for name in binders(field)]
    return []


def substitute(shape, name, value):
    if shape == ("var", name):
        return value
    if shape[0] == "ctr":
        return ("ctr", shape[1], [substitute(field, name, value) for field in shape[2]])
    return shape


def descend(argument, column):
    """The checker's comparison of a self-call argument with its column (term_descend)."""
    if column[0] == "var":
        return "EQ" if argument == column else "GT"
    if column[0] != "ctr":
        return "GT"
    bad = -1
    if argument[0] == "ctr" and argument[1] == column[1] and len(argument[2]) == len(column[2]):
        order = "EQ"
        for index, (field, part) in enumerate(zip(argument[2], column[2])):
            compared = descend(field, part)
            order = order if compared == "EQ" else compared
            if compared == "GT":
                bad = index
                break
        if bad < 0:
            return order
    for index, part in enumerate(column[2]):
        if index != bad and descend(argument, part) != "GT":
            return "LT"
    return "GT"


class Scope:
    """The binders that can still be matched, in order, the current pattern of every column, and what each
    name in the body denotes: a binder (by a unique id), a let value or a rebuilt matched value."""

    def __init__(self, order, columns, env, closed=(), fresh=None):
        self.order, self.columns, self.env, self.closed = list(order), list(columns), dict(env), set(closed)
        self.fresh = fresh if fresh is not None else [0]

    def copy(self):
        return Scope(self.order, self.columns, self.env, self.closed, self.fresh)

    def apply(self, name, pattern):
        target = self.env.get(name)
        shape = term(pattern)
        if target is None or target[0] != "var" or shape == OPAQUE:
            return
        names = binders(shape)
        ids = {}
        for binder in names:
            self.fresh[0] += 1
            ids[binder] = ("var", f"{binder}#{self.fresh[0]}")
        bound = term(pattern, ids)
        self.columns = [substitute(column, target[1], bound) for column in self.columns]
        self.env[name] = bound
        self.env.update(ids)
        if target[1] in self.order:
            position = self.order.index(target[1])
            if bound[0] != "var":
                self.closed.update(self.order[:position])
            self.order[position:position + 1] = binders(bound)

    def is_closed(self, name):
        target = self.env.get(name)
        return target is not None and target[0] == "var" and target[1] in self.closed


def header_end(lines, start):
    """Index of the first body line of a def whose header starts at line start."""
    depth = 0
    for index in range(start, len(lines)):
        for char in lines[index]:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
        if depth == 0 and lines[index].rstrip().endswith(":"):
            return index + 1
    return len(lines)


def erased(text):
    """Positions in types, where the checker does not check self-calls: equations, annotation types, type arguments."""
    mask = [False] * len(text)
    stack = []
    for index, char in enumerate(text):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            start = stack.pop()
            inner = text[start + 1:index]
            if "==" in inner:
                mask[start:index + 1] = [True] * (index + 1 - start)
            else:
                annotated = top_split(inner, " : ")
                if annotated:
                    begin = start + 1 + len(annotated[0])
                    mask[begin:index + 1] = [True] * (index + 1 - begin)
    index = 0
    while index < len(text):
        if text[index] == "<" and index > 0 and (text[index - 1].isalnum() or text[index - 1] in "_.") \
                and text[index + 1:index + 2] != ">":
            depth, end = 1, index + 1
            while end < len(text) and depth:
                if text[end] == "<" and text[end + 1:end + 2] != ">":
                    depth += 1
                elif text[end] == ">" and text[end - 1] not in "-=<":
                    depth -= 1
                end += 1
            mask[index:end] = [True] * (end - index)
            index = end
        else:
            index += 1
    return mask


def law_markers(text):
    """The parameter markers of every law, from its `for` lines."""
    found = {}
    for match in LAW.finditer(text):
        found[match.group(1)] = [line.strip()[4:5] if line.strip()[4:5] in ("+", "-", "~") else ""
                                 for line in match.group(2).splitlines()]
    return found


def case_header(lines, index):
    """The patterns of a case starting on line index, which may continue on later lines."""
    text = lines[index].strip()[5:]
    following = index
    while True:
        split = top_split(text, ":")
        if split is not None or following + 1 >= len(lines):
            return (split[0] if split else text).strip()
        following += 1
        text += " " + lines[following].strip()


def order_and_descent(text, lines, report):
    """A1 (match order) and C4 (decreasing self-calls) for every def in a file."""
    laws = law_markers(text)
    for found in DEF.finditer(text):
        name = found.group(1)
        end = close_paren(text, found.end())
        if end < 0:
            continue
        params = param_names(split_top(text[found.end():end]))
        markers = laws.get(name)
        if markers and len(markers) == len(params):
            params = [(param, marker or law) for (param, marker), law in zip(params, markers)]
        line_end = text.find("\n", end)
        result = text[end + 1:line_end if line_end > 0 else len(text)]
        start = text.count("\n", 0, found.start())
        first = header_end(lines, start)
        last = first
        while last < len(lines) and (not lines[last].strip() or lines[last].startswith((" ", "#"))):
            last += 1
        live = [param for param, marker in params if marker != "~"]
        variables = [("var", param) for param in live]
        stack = [(-1, Scope(live, variables, {param: shape for param, shape in zip(live, variables)}))]
        pending, scopes, depth = [], {}, 0
        for index in range(first, last):
            line = lines[index]
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            opened = depth
            depth += sum(line.count(char) for char in "([{") - sum(line.count(char) for char in ")]}")
            if opened > 0:
                scopes[index] = stack[-1][1]
                continue
            indent = len(line) - len(line.lstrip())
            while len(stack) > 1 and indent <= stack[-1][0]:
                stack.pop()
            pending = [entry for entry in pending if entry[0] < indent]
            scope = stack[-1][1]
            if stripped.startswith("case ") and pending:
                names = pending[-1][1]
                columns = pattern_columns(case_header(lines, index))
                inner = scope.copy()
                if len(columns) == len(names):
                    for scrutinee, pattern in zip(names, columns):
                        inner.apply(scrutinee, pattern)
                stack.append((indent, inner))
                scopes[index] = inner
                continue
            scopes[index] = scope
            header = MATCH.match(line)
            if header:
                names = header.group(1).split()
                for scrutinee in names:
                    if scope.is_closed(scrutinee):
                        report(index + 1, "A1", f"{scrutinee} after a later binder")
                pending.append((indent, names))
                continue
            destructure = DESTRUCTURE.match(line)
            if destructure and not stripped.startswith(("case ", "%")):
                value = destructure.group(3).strip()
                if scope.is_closed(value):
                    report(index + 1, "A1", f"{value} after a later binder")
                scope.apply(value, destructure.group(2))
                continue
            let = re.match(r"^\s*\+?([a-z_]\w*)\s*=(?![=>])\s*(.+)$", line)
            if let and not stripped.startswith(("case ", "%")):
                value = let.group(2).strip()
                balanced = value.count("(") == value.count(")") and value.count("{") == value.count("}")
                scope.env[let.group(1)] = term(value, scope.env) if balanced else OPAQUE
        if re.match(r"^\s*->\s*(Data|Type|Kind)\b", result):
            continue
        body = "\n".join(lines[first:last])
        mask = erased(body)
        for call in re.finditer(r"(?<![\w.~])" + re.escape(name) + r"\(", body):
            if mask[call.start()]:
                continue
            index = first + body.count("\n", 0, call.start())
            while index not in scopes and index > first:
                index -= 1
            scope = scopes.get(index)
            close = close_paren(body, call.end())
            if scope is None or close < 0:
                continue
            args = split_top(body[call.end():close])
            if len(args) != len(params):
                continue
            if any(marker == "~" and arg.lstrip("~").strip() != param
                   for (param, marker), arg in zip(params, args)):
                continue
            order = "EQ"
            column = 0
            for (param, marker), arg in zip(params, args):
                if marker == "~":
                    continue
                if marker != "-":
                    order = descend(term(arg, scope.env), scope.columns[column])
                column += 1
                if order != "EQ":
                    break
            if order != "LT":
                line = first + body.count("\n", 0, call.start()) + 1
                report(line, "C4", f"{name}: {param} becomes {' '.join(arg.split())[:40]}")


TEMPLATE_USE = re.compile(r"~([a-z_]\w*)")
PARAMETER = re.compile(r"[~+-]?([a-z_]\w*)\s*:")
TOP_LEVEL = re.compile(r"^(?:@unsafe\s+def|def|type|law|import)\s", re.M)


def forward_uses(text, report):
    """C5: an unqualified call or template use of a def that the file writes later."""
    starts = [(match.start(), match.group(1)) for match in DEF.finditer(text) if "." not in match.group(1)]
    order = {}
    for index, (_, name) in enumerate(starts):
        order.setdefault(name, index)
    for index, (start, name) in enumerate(starts):
        following = TOP_LEVEL.search(text, start + 1)
        body = text[start:following.start() if following else len(text)]
        header_end = close_paren(body, body.index("(") + 1)
        own = set(PARAMETER.findall(body[:header_end])) if header_end > 0 else set()
        uses = [(match.start(), match.group(1)) for match in CALL.finditer(body) if "." not in match.group(1)]
        uses += [(match.start(), match.group(1)) for match in TEMPLATE_USE.finditer(body)]
        for offset, used in uses:
            if used == name or used in own or order.get(used, -1) <= index:
                continue
            line_start = body.rfind("\n", 0, offset) + 1
            if body[line_start:offset].lstrip().startswith("#"):
                continue
            report(text.count("\n", 0, start + offset) + 1, "C5", f"{name} uses {used}")


def lint(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    findings = []

    def report(line, rule, message):
        findings.append((line, rule, message))

    aliases = {alias: (path.parent / target) for target, alias in IMPORT.findall(text)}
    own = templates(path)
    for name, (leading, late, line) in own.items():
        if late:
            report(line, "C1", f"def {name}")

    for match in CALL.finditer(text):
        end = close_paren(text, match.end())
        if end < 0:
            continue
        args = split_top(text[match.end():end])
        marked = [arg.startswith("~") for arg in args]
        count = sum(marked)
        if count == 0:
            continue
        line = text.count("\n", 0, match.start()) + 1
        if lines[line - 1].lstrip().startswith(("def ", "@unsafe def ")):
            continue
        target = match.group(1)
        if "." in target:
            alias, name = target.split(".", 1)
            table = templates(aliases[alias]) if alias in aliases else None
        else:
            name, table = target, own
        if table is None or name not in table:
            continue
        leading = table[name][0]
        if count > leading or not all(marked[:count]):
            report(line, "C2", f"{target} declares {leading} ~ parameters; the call writes {count}")

    forward_uses(text, report)

    let_indent = None
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if stripped.startswith(("def ", "type ", "law ", "import ", "@unsafe")):
            let_indent = None
        if let_indent is not None and indent < let_indent:
            let_indent = None
        for match in BIG_WORD.finditer(line):
            report(number, "E1", f"U32.to_nat({match.group(1)})")
        if CLOSED_LIMIT.search(line):
            report(number, "E1?", "")
        for match in BIG_NAT.finditer(line):
            report(number, "E1", f"{match.group(1)}n")
        header = MATCH.match(line)
        if header:
            if re.search(r"[({\[]", header.group(1)):
                report(number, "A3", "the scrutinee is an expression")
            elif let_indent is not None:
                report(number, "A2", "a match after a let")
            continue
        destructure = DESTRUCTURE.match(line)
        if destructure and not stripped.startswith(("case ", "%")):
            value = destructure.group(3).strip()
            if not NAME.match(value):
                report(number, "A3", "the destructured value is an expression")
            elif let_indent is not None:
                report(number, "A2", "a destructuring after a let")
            continue
        if LIST_PATTERN.match(line):
            report(number, "A7", "")
        let = LET.match(line)
        if let and not stripped.startswith(("case ", "%", "def ")):
            if BARE_CTOR_LET.match(line):
                report(number, "D4", "")
            if let_indent is None or indent < let_indent:
                let_indent = indent
    order_and_descent(text, lines, report)
    return findings


def main():
    targets = [Path(arg) for arg in sys.argv[1:]]
    if not targets:
        sys.exit("Usage: lint.py FILE_OR_DIR [...]")
    files = []
    for target in targets:
        files.extend(sorted(target.rglob("*.bend")) if target.is_dir() else [target])
    groups = {}
    for path in files:
        for line, rule, detail in sorted(lint(path)):
            groups.setdefault(rule, {}).setdefault(path.as_posix(), []).append((line, detail))
    total = 0
    for rule in sorted(groups):
        found, why, fix = RULES[rule]
        count = sum(len(hits) for hits in groups[rule].values())
        total += count
        print(f"[{rule}] {found}: {count}")
        print(f"  why: {why}")
        print(f"  fix: {fix}")
        for name, hits in groups[rule].items():
            lines = ", ".join(str(line) for line, _ in hits[:12]) + (f" ... ({len(hits)} in all)" if len(hits) > 12 else "")
            print(f"    {name}: line {lines}")
            details = sorted({detail for _, detail in hits if detail})
            if details:
                print(f"      {'; '.join(details[:4])}")
        print()
    print(f"{len(files)} files scanned, {total} findings. Rules: docs/toolchain_rules.md")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
