import re,sys,pathlib
for stem in sys.argv[1:]:
    s=pathlib.Path(f'en/{stem}.qmd').read_text(encoding='utf-8')
    body=re.sub(r'\{\{< include [^>]+>\}\}','',s); body=re.sub(r'<!-- run-links -->.*?<!-- /run-links -->','',body,flags=re.S)
    body=re.sub(r'!\[[^\]]*\]\([^)]*\)','[fig]',body); body=re.sub(r'\n{3,}','\n\n',body)
    i=body.find('## Exercises'); ex=body[i:]; main=body[:i]
    print(f'######## {stem}  ({len(main)} chars)'); print(main[:7000]); print('...'); print(ex[:2500])
