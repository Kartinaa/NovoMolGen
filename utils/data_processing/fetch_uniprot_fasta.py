#!/usr/bin/env python3
"""
Fetch protein sequences from UniProt by accession list and save as FASTA.

Examples:
  python utils/fetch_uniprot_fasta.py \
    --ids /home/yang2531/Documents/Project/Structure_safe/bdnv2_uniprot.txt \
    --out /home/yang2531/Documents/Project/Structure_safe/bdnv2_sequences.fasta

Notes:
  - Uses UniProt REST "stream" endpoint in batches for efficiency
  - Falls back to per-accession fetch for any missing sequences
"""
import argparse
import os
import sys
import time
from typing import Iterable, List

try:
    import requests
except Exception as e:  # pragma: no cover
    print("The 'requests' package is required. Install via: pip install requests", file=sys.stderr)
    raise


UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_ENTRY_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"


def read_accessions(ids_path: str) -> List[str]:
    with open(ids_path, 'r', encoding='utf-8') as f:
        accessions = [line.strip() for line in f if line.strip()]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for acc in accessions:
        if acc not in seen:
            unique.append(acc)
            seen.add(acc)
    return unique


def chunked(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    chunk: List[str] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def fetch_batch_fasta(accessions: List[str], session: requests.Session, retries: int = 3, backoff: float = 1.0) -> str:
    # Query syntax: accession:ACC1 OR accession:ACC2 ...
    query = " OR ".join(f"accession:{acc}" for acc in accessions)
    params = {"format": "fasta", "query": query}
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(UNIPROT_STREAM_URL, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.text
            # 404 indicates none found for the batch; break to fallback per-id
            if resp.status_code == 404:
                break
        except requests.RequestException:
            pass
        time.sleep(backoff * attempt)
    return ""


def fetch_single_fasta(acc: str, session: requests.Session, retries: int = 3, backoff: float = 1.0) -> str:
    url = UNIPROT_ENTRY_FASTA.format(acc=acc)
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200 and resp.text.startswith('>'):
                return resp.text
            if resp.status_code == 404:
                return ""
        except requests.RequestException:
            pass
        time.sleep(backoff * attempt)
    return ""


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as w:
        w.write(text)


def append_text(path: str, text: str) -> None:
    with open(path, 'a', encoding='utf-8') as w:
        w.write(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', required=True, help='Path to txt file with one UniProt accession per line')
    ap.add_argument('--out', required=True, help='Output FASTA path')
    ap.add_argument('--batch-size', type=int, default=200, help='Number of IDs per batch request to UniProt stream')
    ap.add_argument('--sleep', type=float, default=0.2, help='Sleep seconds between requests (be kind to API)')
    args = ap.parse_args()

    accessions = read_accessions(args.ids)
    if not accessions:
        print('No accessions found in input file.', file=sys.stderr)
        sys.exit(1)

    # Start with an empty file
    write_text(args.out, "")

    missing: List[str] = []
    with requests.Session() as session:
        for batch in chunked(accessions, args.batch_size):
            fasta_text = fetch_batch_fasta(batch, session)
            if fasta_text:
                append_text(args.out, fasta_text if fasta_text.endswith('\n') else fasta_text + '\n')
            else:
                missing.extend(batch)
            time.sleep(args.sleep)

        # Fallback per-accession for any that were missing in batch fetch
        if missing:
            still_missing: List[str] = []
            for acc in missing:
                text = fetch_single_fasta(acc, session)
                if text:
                    append_text(args.out, text if text.endswith('\n') else text + '\n')
                else:
                    still_missing.append(acc)
                time.sleep(args.sleep)
            if still_missing:
                missing_path = os.path.splitext(args.out)[0] + '.missing.txt'
                write_text(missing_path, "\n".join(still_missing) + "\n")
                print(f"Warning: {len(still_missing)} accessions not found. See {missing_path}")

    print(f"Wrote FASTA to {args.out}")


if __name__ == '__main__':
    main()


