"""Per-family construction rates over the corpus plus extra sample dirs.

Usage, from the repository root:

    python3 docs/research/frontier-2026-09/measure.py \
        docs/research/frontier-2026-09/patterns.tsv \
        a=docs/research/frontier-2026-09/a b=docs/research/frontier-2026-09/b

Families: `human`, `claude` (manifest model names containing "claude"),
`small` (every other llm model), plus one column per `name=dir` argument.
Fenced code blocks are stripped before counting. Rates are per million
whitespace-separated words; the `d` column counts documents with at least
one match. Only the train and dev splits are read: the sealed holdout
never informs rule design.
"""
import json,re,os,sys,collections
root='corpus'
man={}
for l in open(f'{root}/manifest.jsonl'):
    d=json.loads(l); man[d['id']]=d
def fam(d):
    if d['class']=='human': return 'human'
    m=d['model']['name'] if isinstance(d.get('model'),dict) else d.get('model')
    return 'claude' if 'claude' in m else 'small'
texts=collections.defaultdict(list)
for cls in ('llm','human'):
    for g in os.listdir(f'{root}/{cls}'):
        for f in os.listdir(f'{root}/{cls}/{g}'):
            i=f.rsplit('.',1)[0]
            if i not in man or man[i].get('split') not in ('train', 'dev'): continue
            t=open(f'{root}/{cls}/{g}/{f}',encoding='utf-8',errors='replace').read()
            t=re.sub(r'```.*?```','',t,flags=re.S)
            texts[fam(man[i])].append((g,t))
# extra dirs passed as fam=path
for a in sys.argv[2:]:
    k,p=a.split('=')
    for f in sorted(os.listdir(p)):
        t=open(os.path.join(p,f)).read(); t=re.sub(r'```.*?```','',t,flags=re.S); texts[k].append(('x',t))
pats=[l.rstrip('\n').split('\t',1) for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
fams=list(texts)
words={k:sum(len(t.split()) for _,t in v) for k,v in texts.items()}
print('%-28s'%'pattern'+''.join('%16s'%f'{k}({len(texts[k])}d,{words[k]//1000}k)' for k in fams))
for name,p in pats:
    r=re.compile(p,re.I|re.M)
    row=[]
    for k in fams:
        n=sum(len(r.findall(t)) for _,t in texts[k]); docs=sum(1 for _,t in texts[k] if r.search(t))
        row.append('%6d %5.0f/M %2dd'%(n,n/words[k]*1e6,docs))
    print('%-28s'%name+''.join('%16s'%c for c in row))
