#!/usr/bin/python3

import pathlib
import os
import subprocess
import yaml

FILENAME = 'xmake.yml'
CONFIG = '.git'
OUTDIR = '.out'


def repo_abs_path():
    d = pathlib.Path.cwd()
    while not (d / CONFIG).exists():
        d = d.parent
    return d


def curr_rel_path():
    d = pathlib.Path.cwd()
    return d.relative_to(repo_abs_path())


def prefix_to_rel_path(prefix):
    return prefix.relative_to(pathlib.PurePath('//'))


def prefix_to_abs_path(prefix):
    return repo_abs_path() / prefix_to_rel_path(prefix)


def rel_path_to_prefix(rel_path):
    return pathlib.PurePath('//') / rel_path


def curr_prefix():
    return rel_path_to_prefix(curr_rel_path())


def prefix_to_abs_out_name(prefix):
    abs_path = prefix_to_abs_path(prefix)
    name = abs_path.name
    dep_dir = abs_path.parent
    return dep_dir / OUTDIR / name


def cmd_run(command):
    print(' '.join(command))
    subprocess.run(command)


def do_cpp(name, props):
    CC = 'clang++'
    AR = 'ar'
    LINKER = 'ld'
    OPT = '-O3'
    G = '-g0'
    STD = '-std=c++20'
    INCLUDE = '-I' + str(repo_abs_path())
    PTHREAD = '-pthread'

    if 'main' not in props:
        props['main'] = False
    if 'hdrs' not in props:
        props['hdrs'] = list()
    if 'srcs' not in props:
        props['srcs'] = list()
    srcs = props['srcs']
    hdrs = props['hdrs']
    main = props['main']
    deps = props['deps']

    rule_dir = prefix_to_abs_path(name).parent
    target_out = rule_dir / OUTDIR
    target_out.mkdir(exist_ok=True)

    # The objects we'll link against in this rule. This includes the source
    # objects and the lib files we depend on.
    objects = list()

    # Compile the sources.
    for src in srcs:
        src_path = rule_dir / src
        dot_o = (target_out / src).with_suffix('.o')
        cmd_run([
            CC,
            OPT,
            # G,
            STD,
            INCLUDE,
            '-c',
            str(src_path),
            '-o',
            str(dot_o)
        ])
        objects.append(str(dot_o))

    # Get the compiled dependencies. They should have been compiled outside this
    # function.
    for dep in deps:
        dot_a = prefix_to_abs_out_name(dep).with_suffix('.a')
        objects.append(str(dot_a))

    # Link everything or archive everything.
    out_name = prefix_to_abs_out_name(name)
    if main:
        cmd_run([CC, PTHREAD, '-o', str(out_name)] + objects)

    else:
        cmd_run([AR, 'rcsuUPT', str(out_name.with_suffix('.a'))] + objects)


def load_prefix(prefix):
    print('[Loading]', prefix)
    prop_dict = dict()
    rel_path = prefix_to_rel_path(prefix)
    abs_path = repo_abs_path() / rel_path
    with (abs_path / FILENAME).open() as f:
        rules = yaml.safe_load(f)
        for name, props in rules.items():
            canonical_name = prefix / name

            # Canonicalize deps.
            deps = props['deps'] if 'deps' in props else list()
            dep_paths = (pathlib.PurePath(dep) for dep in deps)
            canonical_deps = (dep_path if dep_path.is_absolute() else prefix /
                              dep_path for dep_path in dep_paths)
            props['deps'] = set(canonical_deps)

            prop_dict[canonical_name] = props
        return prop_dict


loaded_prefixes = set()

prop_dict = load_prefix(curr_prefix())
loaded_prefixes.add(curr_prefix())

targets = set()

size = 0
while size != len(prop_dict):
    size = len(prop_dict)
    new_dict = dict()
    for target, props in prop_dict.items():
        deps = props['deps']
        for dep in deps:
            prefix = dep.parent
            if prefix not in loaded_prefixes:
                new_dict.update(load_prefix(prefix))
                loaded_prefixes.add(prefix)
        targets.add(target)
    prop_dict.update(new_dict)

done_targets = set()
# Nothing fancy. Keep looping over the rules to see which ones have all
# dependencies done. Note names are canonical names already.
while targets != done_targets:
    for target, props in prop_dict.items():
        deps = props['deps']
        if all(dep in done_targets
               for dep in deps) and target not in done_targets:
            print("[Doing]", target)
            do_cpp(target, prop_dict[target])
            done_targets.add(target)
