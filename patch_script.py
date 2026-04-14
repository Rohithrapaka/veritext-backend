from pathlib import Path
text = Path('main.py').read_text()
starts = [i for i in range(len(text)) if text.startswith('Text to analyze:', i)]
if len(starts) < 2:
    raise SystemExit(f'expected at least 2 Text to analyze occurrences, got {len(starts)}')
start = starts[1]
end_marker = '    try:\n        response = requests.post(SAPLING_API_URL, json=payload, timeout=20)'
end = text.index(end_marker, start)
Path('main.py').write_text(text[:start] + text[end:])
print('removed stray helper block')
