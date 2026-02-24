#!/usr/bin/env python3
import pathlib
path = pathlib.Path('.github/workflows/keep-alive.yml')
data = path.read_bytes()
print('len', len(data))
for idx, b in enumerate(data):
    if b > 127:
        print('idx', idx, 'byte', hex(b))
        start = max(0, idx - 20)
        end = min(len(data), idx + 20)
        print('context:', data[start:end])
        break
