from pathlib import Path
p = Path('data/shl_catalog.json')
text = p.read_text(encoding='utf-8')
for i, line in enumerate(text.splitlines(), 1):
    if 4788 <= i <= 4800:
        print(i, repr(line))
