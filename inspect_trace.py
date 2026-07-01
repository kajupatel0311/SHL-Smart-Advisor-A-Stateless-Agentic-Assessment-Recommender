from pathlib import Path
text = Path('conversations/GenAI_SampleConversations/C1.md').read_text(encoding='utf-8')
for line in text.splitlines():
    if line.strip().startswith('|'):
        print(repr(line))
