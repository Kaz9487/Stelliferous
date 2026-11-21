// Where each Bend declaration and file is used: navigation from upstream
// parsing. It lists candidates for review, never what may be deleted, and it
// is not a proof verdict. `bun scripts/bend_dependencies.ts help` lists the
// queries; python run.py dependencies runs the same.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..').replaceAll('\\', '/');
const output = root + '/out/results/dependencies';
const config = JSON.parse(fs.readFileSync(root + '/scripts/dependency_roots.json', 'utf8'));
const SOURCE_DIRS = ['lib', 'convolution', 'examples', 'benchmarks', 'scripts'];
const USAGE = `Usage: python run.py dependencies [QUERY]

  (none)               Summary; writes out/results/dependencies/dependencies.json
  unused [PREFIX]      The review lists, optionally only for files under PREFIX
  why NAME             A shortest chain from a root to NAME
  users NAME           What refers to NAME, or which files import a FILE
  deps NAME            What NAME refers to, or what a FILE imports
  size NAME...         What each NAME reaches, and what it reaches that no other
                       listed NAME does: declarations, files and source lines
  duplicates [PREFIX]  Declarations with the same statement and body, and proof
                       declarations with the same statement and another proof
  --all                With users or deps: also indirectly, each with its distance
  --json               With size or duplicates: print JSON

NAME is a declaration (lib/tensor_select.select, tensor_select.select or a
unique bare name) or a file (lib/tensor_select.bend).`;

const flags = ['--all', '--json'];
const all = process.argv.includes('--all');
const json = process.argv.includes('--json');
const [query, ...queryArguments] = process.argv.slice(2).filter((argument) => !flags.includes(argument));
if (
  (all && !['users', 'deps'].includes(query)) ||
  (json && !['size', 'duplicates'].includes(query)) ||
  (query && !['unused', 'why', 'users', 'deps', 'size', 'duplicates'].includes(query)) ||
  (['why', 'users', 'deps'].includes(query) && queryArguments.length !== 1) ||
  (query === 'size' && !queryArguments.length) ||
  (['unused', 'duplicates'].includes(query) && queryArguments.length > 1)
) {
  const help = ['help', '-h', '--help'].includes(query);
  (help ? console.log : console.error)(USAGE);
  process.exit(help ? 0 : 2);
}
function fail(message: string): never {
  console.error(message);
  process.exit(1);
}

function filesUnder(folder: string, extension: string): string[] {
  if (!fs.existsSync(root + '/' + folder)) return [];
  return fs
    .readdirSync(root + '/' + folder, { recursive: true, encoding: 'utf8' })
    .map((name) => folder + '/' + name.replaceAll('\\', '/'))
    .filter((file) => file.endsWith(extension) && !file.split('/').includes('__pycache__'))
    .sort();
}
const loadedFiles = SOURCE_DIRS.flatMap((folder) => filesUnder(folder, '.bend'));
const texts = new Map<string, string>();
const read = (file: string) => {
  if (!texts.has(file)) texts.set(file, fs.readFileSync(root + '/' + file, 'utf8'));
  return texts.get(file)!;
};

// The parser looks up every import path on disk. Through WSL a checkout on a
// Windows drive answers each lookup slowly, so each source is read once and
// parsed from a copy in the temporary folder. Names are relative to the copy,
// so the report is the same.
const stage = fs.mkdtempSync(os.tmpdir() + '/bend-dependencies-').replaceAll('\\', '/');
process.on('exit', () => fs.rmSync(stage, { recursive: true, force: true }));
for (const file of loadedFiles) {
  fs.mkdirSync(path.posix.dirname(stage + '/' + file), { recursive: true });
  fs.writeFileSync(stage + '/' + file, read(file));
}

// Examples import the published package. Serve each configured package from
// its folder in this checkout, so the report runs offline and the examples'
// uses count against the checkout's own declarations. The package is a link
// to the folder: upstream resolves it to the real file, already loaded under
// the folder's namespace, and parses it once.
const mirror = stage + '/.bend_lib';
const packageImports = new Set(
  loadedFiles.flatMap((file) =>
    [...read(file).matchAll(/^import\s+([a-z][a-z0-9-]*@[0-9.]+)\//gm)].map((match) => match[1]),
  ),
);
for (const [index, name] of Object.keys(config.packages).entries()) {
  const hash = '0x' + (index + 1).toString(16).padStart(32, '0');
  fs.mkdirSync(mirror + '/names', { recursive: true });
  fs.symlinkSync(stage + '/' + config.packages[name], mirror + '/' + hash, 'dir');
  for (const imported of packageImports) {
    if (imported.startsWith(name + '@')) fs.writeFileSync(mirror + '/names/' + imported, hash + '\n');
  }
}
process.env.BEND_LIB = mirror;
const { book_nil, book_load, term_lower, err_show } = await import('../.tools/bend/bend2/bend.ts');

// A namespace is a file's path from the repository root. An internal key is
// namespace:name; the report spells it as the file's path without .bend, a dot
// and the name.
const fileOfNamespace = (namespace: string) => namespace + '.bend';
function keyParts(key: string): { file: string; bare: string; name: string } {
  const at = key.indexOf(':');
  const file = fileOfNamespace(key.slice(0, at));
  const bare = key.slice(at + 1);
  return { file, bare, name: file.slice(0, -5) + '.' + bare };
}

// Every file loads against the repository root, so one file keeps one
// namespace whichever entry reaches it first.
const book = book_nil();
const seen = new Map<string, string | null>();
const failures: { file: string; error: string }[] = [];
for (const file of loadedFiles) {
  try {
    await book_load(book, stage + '/' + file, file.slice(0, -5), seen, undefined, stage + '/');
  } catch (error: any) {
    const message = error?.$ === 'Err' ? err_show(error) : String(error?.message ?? error);
    // The message without colour codes, up to the source excerpt.
    const lines = message
      .replace(/\x1b\[[0-9;]*m/g, '')
      .split('\n')
      .map((line: string) => line.trim())
      .filter((line: string) => line && line !== 'Error:');
    failures.push({ file, error: lines.slice(0, 4).join(' | ') });
    // A failed load leaves its files marked as in progress.
    for (const [loaded, namespace] of seen) if (namespace === null) seen.delete(loaded);
  }
}

// Declarations: references from the type and the body, including all
// branches; a constructor counts as its datatype. Base is left out. A law's
// body may be written in another file, which then uses what the body names.
type Declaration = {
  file: string;
  line: number;
  lines: number;
  kind: string;
  signature: Set<string>;
  body: Set<string>;
  body_file: string;
  references: Set<string>;
  statement: string;
  definition: string;
  nodes: number;
  equation: boolean;
};
const declarations = new Map<string, Declaration>();
// Where each name is first stated in its file, with which keyword, and how
// many lines it spans up to the next statement.
const headings = new Map<string, Map<string, [number, string, number]>>();
function heading(file: string, bare: string): [number, string, number] | undefined {
  if (!headings.has(file)) {
    const found = new Map<string, [number, string, number]>();
    const lines = texts.has(file) || fs.existsSync(root + '/' + file) ? read(file).split('\n') : [];
    const starts: [string, number, string][] = [];
    for (const [index, line] of lines.entries()) {
      const match = /^(def|type|law)\s+([\w.]+)/.exec(line);
      if (match) starts.push([match[2], index + 1, match[1]]);
    }
    for (const [index, [bare, line, keyword]] of starts.entries()) {
      const end = index + 1 < starts.length ? starts[index + 1][1] : lines.length + 1;
      if (!found.has(bare)) found.set(bare, [line, keyword, end - line]);
    }
    headings.set(file, found);
  }
  return headings.get(file)!.get(bare);
}

// A structural fingerprint of a term: the same shape and the same referenced
// declarations give the same digest, whatever the bound variables are called.
// A reference to the declaration itself is written as one mark, so a copy that
// recurses on itself under another name matches.
const BINDS = new Set(['Var', 'Lam', 'All', 'Let']);
const NAMED = new Set(['Ref', 'ADT', 'Ctr', 'Mat']);
const shapes = new Map<string, { name: string; statement: Skeleton; whole: Skeleton }>();
const families = new Map<string, string>();
// The declaration a key stands for once copies are merged; a constructor
// stands for its datatype's representative and its bare name.
const canon = new Map<string, string>();
function representative(key: string): string {
  while (canon.has(key)) key = canon.get(key)!;
  return key;
}
function canonical(key: string): string {
  if (book.ctrs[key]) {
    if (!families.has(key)) families.set(key, book.ctrs[key].f);
    return representative(families.get(key)!) + '#' + key.slice(key.indexOf(':') + 1);
  }
  return representative(key);
}
// The shape is walked once: the digest of everything but the names it refers
// to, and those names in order. A round of merging only renames them.
type Skeleton = { shape: string; names: string[]; nodes: number; equation: boolean };
function skeleton(terms: unknown[]): Skeleton {
  const tokens: string[] = [];
  const names: string[] = [];
  let nodes = 0;
  let equation = false;
  const pending: unknown[] = [...terms].reverse();
  while (pending.length) {
    const node: any = pending.pop();
    if (node === null || typeof node !== 'object') {
      tokens.push(typeof node + ':' + String(node));
      continue;
    }
    if (Array.isArray(node)) {
      tokens.push('[' + node.length);
      for (let index = node.length - 1; index >= 0; index--) pending.push(node[index]);
      continue;
    }
    nodes++;
    if (node.$ === 'Eql') equation = true;
    tokens.push('{' + node.$);
    const keys = Object.keys(node).filter((key) => key !== 's' && key !== '$' && !(key === 'k' && BINDS.has(node.$))).sort();
    for (let index = keys.length - 1; index >= 0; index--) {
      const key = keys[index];
      if (key === 'k' && NAMED.has(node.$) && typeof node.k === 'string') {
        names.push(node.k);
        pending.push('#name');
      } else pending.push(node[key]);
      pending.push(key);
    }
  }
  return { shape: createHash('sha1').update(tokens.join(';')).digest('hex'), names, nodes, equation };
}
function fingerprint(part: Skeleton, self: string): string {
  const own = canonical(self);
  const named = part.names.map((name) => (name === self || canonical(name) === own ? '@self' : canonical(name)));
  return createHash('sha1').update(part.shape + '|' + named.join(',')).digest('hex');
}
function walk(term: unknown, visit: (node: any) => boolean) {
  const pending: any[] = [term];
  while (pending.length) {
    const node = pending.pop();
    if (!node || typeof node !== 'object') continue;
    if (visit(node)) return;
    for (const key in node) if (key !== 's') pending.push(node[key]);
  }
}
function references(term: unknown, into: Set<string>) {
  walk(term, (node) => {
    if (['Ref', 'ADT', 'Ctr', 'Mat'].includes(node.$)) {
      for (const key of [node.k, book.ctrs[node.k]?.f ?? null]) {
        if (key && key in book.tlds && key.includes(':')) into.add(keyParts(key).name);
      }
    }
    return false;
  });
}
for (const [key, term] of Object.entries(book.tlds) as [string, any][]) {
  if (!key.includes(':')) continue;
  const { file, bare, name } = keyParts(key);
  const [line, keyword, lines] = heading(file, bare) ?? [0, 'def', 0];
  const type = term_lower(term.T);
  const declaration: Declaration = {
    file,
    line,
    lines,
    kind: term.$ === 'ADT' ? 'type' : keyword,
    signature: new Set(),
    body: new Set(),
    body_file: file,
    references: new Set(),
    statement: '',
    definition: '',
    nodes: 0,
    equation: false,
  };
  references(type, declaration.signature);
  let parts: unknown[];
  if (term.$ === 'Def') {
    const body = term.e ?? (term.v && term_lower(term.v));
    references(body, declaration.body);
    walk(body, (node) => {
      const namespace = node.s?.file?.ns;
      if (typeof namespace === 'string' && namespace) declaration.body_file = fileOfNamespace(namespace);
      return typeof namespace === 'string';
    });
    parts = [body];
  } else {
    // Constructors are compared by their bare names and fields.
    parts = term.c.map((constructor: any) => [constructor.k.slice(constructor.k.indexOf(':') + 1), term_lower(constructor.T)]);
    for (const constructor of term.c) references(term_lower(constructor.T), declaration.signature);
  }
  const statement = skeleton([type]);
  const whole = skeleton([type, parts]);
  shapes.set(key, { name, statement, whole });
  declaration.nodes = whole.nodes;
  // A theorem: after its parameters it states an equation, so any proof of it
  // serves. A declaration whose result is a type or data is not one.
  let result: any = type;
  while (result && result.$ === 'All') result = result.B;
  declaration.equation = result?.$ === 'Eql';
  for (const target of [...declaration.signature, ...declaration.body]) {
    if (target !== name) declaration.references.add(target);
  }
  declarations.set(name, declaration);
}
/// Merge copies until no new pair appears: a copy whose helpers are copies too
// is found in a later round. Only the summary and the duplicates query need it.
let merged = false;
function mergeCopies() {
  if (merged) return;
  merged = true;
  for (let round = 0; round < 8; round++) {
    const groups = new Map<string, string[]>();
    for (const [key, shape] of shapes) {
      const digest = fingerprint(shape.whole, key);
      declarations.get(shape.name)!.definition = digest;
      if (groups.has(digest)) groups.get(digest)!.push(key);
      else groups.set(digest, [key]);
    }
    let changed = false;
    for (const keys of groups.values()) {
      if (keys.length < 2) continue;
      const chosen = keys.map(representative).sort()[0];
      for (const key of keys) {
        if (representative(key) !== chosen) {
          canon.set(representative(key), chosen);
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
  for (const [key, shape] of shapes) declarations.get(shape.name)!.statement = fingerprint(shape.statement, key);
}
const users = new Map<string, Set<string>>([...declarations.keys()].map((name) => [name, new Set()]));
for (const [name, declaration] of declarations) {
  for (const target of declaration.references) users.get(target)!.add(name);
}
const declared = new Map<string, string[]>();
for (const [name, declaration] of declarations) {
  declared.set(declaration.file, [...(declared.get(declaration.file) ?? []), name]);
}

// Files: what each one imports, and which imports its declarations use.
type Import = { alias: string; file: string; line: number };
const imports = new Map<string, Import[]>();
for (const file of loadedFiles) {
  const list: Import[] = [];
  for (const [index, text] of read(file).split('\n').entries()) {
    const line = text.trim();
    if (!line || line.startsWith('#')) continue;
    if (!/^import(\s|$)/.test(line)) break;
    const match = /^import\s+(\S+)\s+as\s+(\w+)/.exec(line);
    if (!match) continue;
    const packaged = /^([a-z][a-z0-9-]*)@[^/]*\/(.*)$/.exec(match[1]);
    const target = packaged
      ? path.posix.join(config.packages[packaged[1]] ?? packaged[1], packaged[2])
      : path.posix.normalize(path.posix.join(path.posix.dirname(file), match[1]));
    list.push({ alias: match[2], file: target, line: index + 1 });
  }
  imports.set(file, list);
}
const fileUses = new Map<string, Set<string>>();
function use(file: string, target: string) {
  const owner = declarations.get(target)!.file;
  if (owner !== file) fileUses.set(file, (fileUses.get(file) ?? new Set()).add(owner));
}
// A file that writes the body of a law another file states is needed by its
// importers although they name nothing in it: importing it loads that body.
const fillsLaws = new Set<string>();
for (const [name, declaration] of declarations) {
  for (const target of declaration.signature) if (target !== name) use(declaration.file, target);
  for (const target of declaration.body) if (target !== name) use(declaration.body_file, target);
  if (declaration.body_file !== declaration.file) {
    use(declaration.body_file, name);
    fillsLaws.add(declaration.body_file);
  }
}
const loadFailed = new Set(failures.map((failure) => failure.file));
// Frozen files keep their bytes; an unused import there stays.
const frozen = new Set(Object.keys(JSON.parse(read('data/contracts/frozen.json'))));
const unusedImports: { file: string; line: number; alias: string; imports: string; frozen: boolean }[] = [];
for (const [file, list] of imports) {
  // A file without declarations imports only to load, as a checker entry does.
  if (!declared.has(file) || loadFailed.has(file)) continue;
  for (const item of list) {
    if (!fileUses.get(file)?.has(item.file) && !fillsLaws.has(item.file)) {
      unusedImports.push({
        file,
        line: item.line,
        alias: item.alias,
        imports: item.file,
        frozen: frozen.has(file),
      });
    }
  }
}

// Roots. Program roots: the configured public API and entries, the main of
// every program outside the library, and the declarations that the Bend
// programs written by Python checks name. Proof roots: the configured
// theorems, and every proof declaration that nothing refers to, a result.
type Root = { name: string; reason: string };
function configuredRoots(entries: { file: string; names: string[] | '*'; reason: string }[]): Root[] {
  return entries.flatMap((entry) => {
    const names =
      entry.names === '*'
        ? (declared.get(entry.file) ?? [])
        : entry.names.map((name) => entry.file.slice(0, -5) + '.' + name);
    if (!names.length || names.some((name) => !declarations.has(name))) {
      for (const failure of failures) console.error(`Could not load ${failure.file}: ${failure.error}`);
      fail(`A configured root is missing: ${entry.file} ${JSON.stringify(entry.names)}`);
    }
    return names.map((name) => ({ name, reason: entry.reason }));
  });
}
const isProofFile = (file: string) => file.split('/').includes('proofs');
const writtenPrograms: Root[] = [];
for (const file of ['benchmarks', 'examples', 'scripts'].flatMap((folder) => filesUnder(folder, '.py'))) {
  const text = read(file);
  for (const [, module, alias] of text.matchAll(
    /import\s+(?:\.\.\/)+((?:lib|convolution|benchmarks|examples)\/[\w/]+)\.bend\s+as\s+(\w+)/g,
  )) {
    for (const [, bare] of text.matchAll(new RegExp(`\\b${alias}\\.(\\w+)`, 'g'))) {
      const name = module + '.' + bare;
      if (declarations.has(name) && !writtenPrograms.some((item) => item.name === name)) {
        writtenPrograms.push({ name, reason: `Named by the Bend program ${file} writes` });
      }
    }
  }
}
const programRoots = [
  ...configuredRoots(config.program),
  ...[...declarations.keys()]
    .filter((name) => name.endsWith('.main') && !/^(lib|convolution)\//.test(name))
    .map((name) => ({ name, reason: 'Program entry' })),
  ...writtenPrograms,
];
const theoremRoots = configuredRoots(config.proof);
const results = [...declarations.keys()]
  .filter((name) => isProofFile(declarations.get(name)!.file) && !users.get(name)!.size)
  .filter((name) => !theoremRoots.some((item) => item.name === name))
  .sort();
const proofRoots = [
  ...theoremRoots,
  ...results.map((name) => ({ name, reason: 'Proof result: nothing refers to it' })),
];

// Breadth first, so each reached declaration remembers a shortest way in.
function reach(roots: string[]): Map<string, string | null> {
  const parent = new Map<string, string | null>(roots.map((name) => [name, null]));
  const queue = [...parent.keys()];
  for (let index = 0; index < queue.length; index++) {
    for (const target of [...declarations.get(queue[index])!.references].sort()) {
      if (parent.has(target)) continue;
      parent.set(target, queue[index]);
      queue.push(target);
    }
  }
  return parent;
}
const programReach = reach(programRoots.map((item) => item.name));
const proofReach = reach(proofRoots.map((item) => item.name));
function status(name: string): string {
  if (programReach.has(name)) return 'program';
  if (proofReach.has(name)) return 'proof';
  return 'unreached';
}
// What each proof result alone keeps: reached from it, and from no program
// root, configured theorem or other result. Deleting the result frees these.
const shared = reach([...programRoots, ...theoremRoots].map((item) => item.name));
const keptBy = new Map<string, string[]>();
for (const result of results) {
  for (const name of reach([result]).keys()) {
    if (name !== result && !shared.has(name)) keptBy.set(name, [...(keptBy.get(name) ?? []), result]);
  }
}
const alone = new Map<string, number>(results.map((name) => [name, 0]));
for (const owners of keptBy.values())
  if (owners.length === 1) alone.set(owners[0], alone.get(owners[0])! + 1);

const describe = (name: string) => {
  const item = declarations.get(name)!;
  return `${name}  ${item.kind} ${item.file}:${item.line} [${status(name)}]`;
};

// Copies: the same statement and body once copies are merged, at least
// MINIMUM_NODES terms, so a one-line accessor alone does not count, and the
// smaller copies a listed group needs. Repeated statements: proof
// declarations stating the same equation with another proof. Each group lists
// its declarations by name.
const MINIMUM_NODES = 40;
function duplicates(prefix: string) {
  mergeCopies();
  const listed = (name: string) => declarations.get(name)!.file.startsWith(prefix);
  // The lines a group would free if one member were kept.
  const lines = (names: string[]) =>
    names.reduce((sum, name) => sum + declarations.get(name)!.lines, 0) -
    Math.max(...names.map((name) => declarations.get(name)!.lines));
  const order = (groups: string[][]) =>
    groups.map((names) => names.sort()).sort((a, b) => lines(b) - lines(a) || a[0].localeCompare(b[0]));
  // Every merged copy, by its representative, whatever its size.
  const members = new Map<string, string[]>();
  const keyOf = new Map<string, string>();
  for (const [key, shape] of shapes) {
    keyOf.set(shape.name, key);
    const chosen = representative(key);
    members.set(chosen, [...(members.get(chosen) ?? []), shape.name]);
  }
  const groupOf = (name: string) => {
    const key = keyOf.get(name);
    const names = key ? members.get(representative(key))! : [name];
    return names.length > 1 ? names : null;
  };
  const copies = [...members.values()].filter((names) => names.length > 1 && names.some((name) => declarations.get(name)!.nodes >= MINIMUM_NODES));
  const copied = new Set(copies.flat());
  const found = new Map<string, string[]>();
  for (const [name, item] of declarations) {
    if (isProofFile(item.file) && item.equation && item.kind !== 'type' && !copied.has(name)) {
      found.set(item.statement, [...(found.get(item.statement) ?? []), name]);
    }
  }
  const statements = [...found.values()].filter((names) => names.length > 1);
  // A reported group is the same once smaller copies count as one, so those
  // copies are reported too: each listed group can be merged as it stands.
  const reported = new Set(copied);
  const pending = [...copies.flat(), ...statements.flat()];
  while (pending.length) {
    for (const target of declarations.get(pending.pop()!)!.references) {
      const group = groupOf(target);
      if (group && !reported.has(target)) {
        copies.push(group);
        for (const name of group) {
          reported.add(name);
          pending.push(name);
        }
      }
    }
  }
  const keep = (groups: string[][]) => order(groups.filter((names) => names.some(listed)));
  return {
    copies: keep(copies).map((names) => ({ lines: lines(names), names })),
    statements: keep(statements).map((names) => ({ lines: lines(names), names })),
  };
}

// What each name reaches, with the part that no other listed name reaches.
function sizes(names: string[]) {
  const reached = new Map(names.map((name) => [name, reach([name])]));
  const count = new Map<string, number>();
  for (const set of reached.values()) for (const item of set.keys()) count.set(item, (count.get(item) ?? 0) + 1);
  const measure = (items: string[]) => {
    const proof = items.filter((item) => isProofFile(declarations.get(item)!.file));
    return {
      declarations: items.length,
      proof_declarations: proof.length,
      files: new Set(items.map((item) => declarations.get(item)!.file)).size,
      proof_files: new Set(proof.map((item) => declarations.get(item)!.file)).size,
      proof_lines: proof.reduce((sum, item) => sum + declarations.get(item)!.lines, 0),
    };
  };
  return names.map((name) => {
    const items = [...reached.get(name)!.keys()];
    return { name, reaches: measure(items), alone: measure(items.filter((item) => count.get(item) === 1)) };
  });
}
const buildReport = () => ({
  scope:
    'Conservative source-term references from the upstream parser, including types and all branches. Not a runtime call graph, a minimal proof dependency set or a checker verdict. Every candidate needs human review before deletion.',
  statuses: {
    program: 'A program root reaches it',
    proof: 'Only a proof root reaches it',
    unreached: 'No root reaches it',
  },
  roots: { program: programRoots, proof: theoremRoots },
  proof_results: Object.fromEntries(results.map((name) => [name, { keeps_alone: alone.get(name)! }])),
  duplicates: duplicates(''),
  load_failures: failures,
  unused_imports: unusedImports,
  files: Object.fromEntries(
    [...imports.keys()].sort().map((file) => {
      const counts: Record<string, number> = {};
      for (const name of declared.get(file) ?? []) counts[status(name)] = (counts[status(name)] ?? 0) + 1;
      return [
        file,
        {
          declarations: counts,
          imports: imports.get(file)!.map((item) => item.file),
          imported_by: [...imports]
            .filter(([, list]) => list.some((item) => item.file === file))
            .map(([by]) => by),
        },
      ];
    }),
  ),
  declarations: Object.fromEntries(
    [...declarations.keys()].sort().map((name) => {
      const item = declarations.get(name)!;
      return [
        name,
        {
          file: item.file,
          line: item.line,
          lines: item.lines,
          kind: item.kind,
          ...(item.body_file !== item.file ? { body_file: item.body_file } : {}),
          status: status(name),
          references: [...item.references].sort(),
          users: [...users.get(name)!].sort(),
        },
      ];
    }),
  ),
});

// Which declarations a name means, or null for a file.
function resolve(text: string): string[] | null {
  const name = text.replace(/^\.\//, '').replace(/\.bend:/, '.');
  if (name.endsWith('.bend')) {
    if (!imports.has(name)) fail(`Not a loaded file: ${text}`);
    return null;
  }
  if (declarations.has(name)) return [name];
  const matches = [...declarations.keys()].filter(
    (key) => key.endsWith('/' + name) || key.endsWith('.' + name),
  );
  if (!matches.length) fail(`No declaration matches ${text}`);
  return matches;
}
function single(text: string): string {
  const matches = resolve(text);
  if (matches === null) fail(`Give a declaration, not a file: ${text}`);
  if (matches.length > 1) fail(`${text} is ambiguous:\n  ${matches.map(describe).join('\n  ')}`);
  return matches[0];
}

// Breadth first from one start: everything next reaches, with its distance.
function distances(start: string, next: (item: string) => Iterable<string>): Map<string, number> {
  const distance = new Map([[start, 0]]);
  const queue = [start];
  for (let index = 0; index < queue.length; index++) {
    for (const item of next(queue[index])) {
      if (distance.has(item)) continue;
      distance.set(item, distance.get(queue[index])! + 1);
      queue.push(item);
    }
  }
  distance.delete(start);
  return distance;
}
const importers = new Map<string, string[]>();
for (const [by, list] of imports) {
  for (const item of list) importers.set(item.file, [...(importers.get(item.file) ?? []), by]);
}
// The direct ones as before; with --all, every one with its distance.
function listing(start: string, next: (item: string) => Iterable<string>, show: (item: string) => string) {
  const found = all ? distances(start, next) : new Map([...next(start)].map((item) => [item, 1]));
  const lines = [...found]
    .sort(([a, x], [b, y]) => x - y || a.localeCompare(b))
    .map(([item, distance]) => (all ? `[${distance}] ` : '') + show(item));
  return lines;
}

if (query === 'why') {
  const name = single(queryArguments[0]);
  const reached = programReach.has(name) ? programReach : proofReach.has(name) ? proofReach : null;
  if (reached === null) {
    console.log(describe(name));
    console.log(
      `No root reaches it. Referred to by: ${[...users.get(name)!].sort().join(', ') || 'nothing'}`,
    );
  } else {
    const chain = [name];
    while (reached.get(chain[0])) chain.unshift(reached.get(chain[0])!);
    const start = [...programRoots, ...proofRoots].find((item) => item.name === chain[0])!;
    console.log(`From the root ${chain[0]} (${start.reason}):`);
    for (const [depth, item] of chain.entries()) console.log('  '.repeat(depth + 1) + describe(item));
  }
} else if (query === 'users' || query === 'deps') {
  const matches = resolve(queryArguments[0]);
  if (matches === null) {
    const file = queryArguments[0].replace(/^\.\//, '');
    const used = (by: string, target: string) => fileUses.get(by)?.has(target) || fillsLaws.has(target);
    const direct = (item: string) => imports.get(file)!.find((entry) => entry.file === item);
    const lines =
      query === 'deps'
        ? listing(
            file,
            (item) => (imports.get(item) ?? []).map((entry) => entry.file),
            (item) =>
              direct(item)
                ? `${item}  as ${direct(item)!.alias}${used(file, item) ? '' : '  (unused)'}`
                : item,
          )
        : listing(
            file,
            (item) => importers.get(item) ?? [],
            (item) =>
              imports.get(item)!.some((entry) => entry.file === file) && !used(item, file)
                ? `${item}  (unused import)`
                : item,
          );
    console.log(
      `${file} ${query === 'deps' ? 'imports' : 'is imported by'} ${lines.length} files${all ? ', directly or not' : ''}:`,
    );
    for (const line of lines) console.log('  ' + line);
  } else {
    if (matches.length > 1)
      fail(`${queryArguments[0]} is ambiguous:\n  ${matches.map(describe).join('\n  ')}`);
    const name = matches[0];
    const lines = listing(
      name,
      (item) => (query === 'deps' ? declarations.get(item)!.references : users.get(item)!),
      describe,
    );
    console.log(describe(name));
    console.log(
      `${query === 'deps' ? 'Refers to' : 'Referred to by'} ${lines.length}${all ? ', directly or not' : ''}:`,
    );
    for (const line of lines) console.log('  ' + line);
  }
} else if (query === 'size') {
  const found = sizes(queryArguments.map(single));
  if (json) console.log(JSON.stringify(found, null, 2));
  else {
    const text = (part: ReturnType<typeof sizes>[number]['reaches']) =>
      `${part.declarations} declarations in ${part.files} files, of them ${part.proof_declarations} in ${part.proof_files} proof files, ${part.proof_lines} proof lines`;
    for (const item of found) {
      console.log(describe(item.name));
      console.log('  reaches ' + text(item.reaches));
      if (found.length > 1) console.log('  alone   ' + text(item.alone));
    }
  }
} else if (query === 'duplicates') {
  const found = duplicates((queryArguments[0] ?? '').replace(/^\.\//, ''));
  if (json) console.log(JSON.stringify(found, null, 2));
  else {
    for (const [title, groups] of [
      ['Copies: the same statement and body', found.copies],
      ['Repeated statements: the same equation proved again', found.statements],
    ] as const) {
      console.log(`${title} (${groups.length} groups, ${groups.reduce((sum, item) => sum + item.lines, 0)} lines beyond one of each):`);
      for (const item of groups) {
        console.log(`  ${item.lines} lines`);
        for (const name of item.names) console.log('    ' + describe(name));
      }
    }
  }
} else {
  if (!query) {
    fs.mkdirSync(output, { recursive: true });
    fs.writeFileSync(output + '/dependencies.json', JSON.stringify(buildReport(), null, 2) + '\n');
  }
  const prefix = (queryArguments[0] ?? '').replace(/^\.\//, '');
  const listed = (name: string) => declarations.get(name)!.file.startsWith(prefix);
  const program = [...declarations.keys()].filter(
    (name) => !isProofFile(declarations.get(name)!.file) && listed(name),
  );
  const referredBy = (name: string) =>
    `, ${users.get(name)!.size ? `${users.get(name)!.size} referrers` : 'nothing refers to it'}`;
  const sections: [string, string[]][] = [
    [
      'Unused imports',
      unusedImports
        .filter((item) => item.file.startsWith(prefix))
        .map(
          (item) =>
            `${item.file}:${item.line}  ${item.alias} -> ${item.imports}${item.frozen ? '  (frozen file)' : ''}`,
        ),
    ],
    [
      'Library and program declarations only proofs reach',
      program.filter((name) => status(name) === 'proof').map(describe),
    ],
    [
      'Declarations no root reaches',
      [...declarations.keys()]
        .filter((name) => listed(name) && status(name) === 'unreached')
        .map((name) => describe(name) + referredBy(name)),
    ],
    [
      'Proof results, nothing refers to them; each keeps the count alone',
      results.filter(listed).map((name) => `${describe(name)}, keeps ${alone.get(name)!}`),
    ],
  ];
  if (query === 'unused') {
    for (const [title, lines] of sections) {
      console.log(`${title} (${lines.length}):`);
      for (const line of lines.sort()) console.log('  ' + line);
    }
  } else {
    const counts = ['program', 'proof', 'unreached'].map(
      (group) => `${[...declarations.keys()].filter((name) => status(name) === group).length} ${group}`,
    );
    console.log(`${declarations.size} declarations in ${declared.size} files: ${counts.join(', ')}.`);
    for (const [title, lines] of sections) console.log(`  ${title}: ${lines.length}`);
    const copies = duplicates('');
    console.log(`  Copied declarations: ${copies.copies.length} groups, repeated statements: ${copies.statements.length} groups (the duplicates query lists them)`);
    for (const failure of failures) console.log(`Could not load ${failure.file}: ${failure.error}`);
    console.log('Report: out/results/dependencies/dependencies.json. The unused query lists the candidates.');
  }
}
