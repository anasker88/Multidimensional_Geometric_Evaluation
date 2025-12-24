#!/usr/bin/env python3
"""Augment `data/questions.csv` by permuting vertex labels.

Generates `data/questions_augmented.csv` with extra columns `aug_source_id` and `aug_map`.
"""
import csv
import re
import random
import itertools
from pathlib import Path


IN = Path(__file__).resolve().parents[1] / "data" / "questions.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "questions_augmented.csv"
RESERVED = set(list("OIl0"))


def find_letters_in_row(row):
    s = " ".join([row.get('2D','') or '', row.get('3D','') or '', row.get('4D','') or ''])
    letters = sorted(set(re.findall(r"[A-Z]", s)))
    letters = [c for c in letters if c not in RESERVED]
    return letters


def make_mappings(letters, n=5, seed=None):
    if not letters:
        return []
    letters = list(letters)
    mappings = []
    L = len(letters)
    # rotations
    for shift in range(1, min(n, L)):
        m = {letters[i]: letters[(i+shift) % L] for i in range(L)}
        mappings.append(m)
    # simple swaps
    for a, b in itertools.combinations(range(L), 2):
        if len(mappings) >= n:
            break
        m = {c: c for c in letters}
        m[letters[a]] = letters[b]
        m[letters[b]] = letters[a]
        mappings.append(m)
    # random perms
    rnd = random.Random(seed)
    tries = 0
    while len(mappings) < n and tries < 50:
        perm = letters[:]
        rnd.shuffle(perm)
        m = {letters[i]: perm[i] for i in range(L)}
        # skip identity
        if all(m[k] == k for k in m):
            tries += 1
            continue
        # avoid duplicate mapping
        if any(all(mp.get(k) == m.get(k) for k in letters) for mp in mappings):
            tries += 1
            continue
        mappings.append(m)
        tries += 1
    return mappings


def apply_map_to_text(text, mapping):
    if not text:
        return text

    # replace sequences of uppercase letters first (e.g. ABEF)
    def seq_repl(mo):
        s = mo.group(0)
        return ''.join(mapping.get(ch, ch) for ch in s)

    text = re.sub(r"[A-Z]{2,}", seq_repl, text)
    # replace isolated single-letter tokens
    text = re.sub(r"\b([A-Z])\b", lambda mo: mapping.get(mo.group(1), mo.group(1)), text)
    return text


def augment(in_path: Path, out_path: Path, n_maps_per_row: int = 5):
    seen = set()
    out_rows = []
    with in_path.open(newline='', encoding='utf-8') as inf:
        reader = csv.DictReader(inf)
        headers = reader.fieldnames[:] if reader.fieldnames else ['2D','3D','4D','answer','type']
        # add augmentation meta fields
        headers += ['aug_source_id', 'aug_map']
        for i, row in enumerate(reader):
            # normalize keys
            r = {k: (row.get(k) or '') for k in headers if k not in ('aug_source_id','aug_map')}
            key = (r.get('2D',''), r.get('3D',''), r.get('4D',''), r.get('answer',''), r.get('type',''))
            if key not in seen:
                seen.add(key)
                base = dict(r)
                base['aug_source_id'] = i
                base['aug_map'] = ''
                out_rows.append(base)

            letters = find_letters_in_row(r)
            mappings = make_mappings(letters, n=n_maps_per_row, seed=i)
            for m in mappings:
                newrow = {}
                for col in ['2D','3D','4D','answer','type']:
                    val = r.get(col, '')
                    if col in ('2D','3D','4D') and val:
                        val2 = apply_map_to_text(val, m)
                    else:
                        val2 = val
                    newrow[col] = val2
                key2 = (newrow.get('2D',''), newrow.get('3D',''), newrow.get('4D',''), newrow.get('answer',''), newrow.get('type',''))
                if key2 in seen:
                    continue
                seen.add(key2)
                newrow['aug_source_id'] = i
                newrow['aug_map'] = ','.join(f"{k}->{v}" for k, v in sorted(m.items()))
                out_rows.append(newrow)

    # write out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as outf:
        writer = csv.DictWriter(outf, fieldnames=headers)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r.get(k, '') for k in headers})

    return len(out_rows), len(out_rows) - sum(1 for _ in open(in_path, encoding='utf-8'))


if __name__ == '__main__':
    print(f"Reading: {IN}")
    n_total, n_added = augment(IN, OUT, n_maps_per_row=5)
    print(f"Wrote: {OUT} (total rows: {n_total}, added: {n_added})")
