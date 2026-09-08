"""Local technical validation; intentionally never opens rendered PNGs."""
from pathlib import Path
from collections import Counter
import json
import re
import shutil
import subprocess
import zipfile
import hashlib

root = Path('D:/project/my/research/streamllm_p2')
sc = root / 'paper2/tougao/SC'
src = sc / 'latex'
v = Path(__file__).resolve().parent

def run(cmd, cwd, name):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    (v / name).write_bytes(p.stdout)
    return p.stdout.decode(errors='replace')

def expand(path):
    text = path.read_text(encoding='utf-8')
    def load(m):
        q = src / m[1]
        if not q.suffix:
            q = q.with_suffix('.tex')
        return expand(q)
    return re.sub(r'\\(?:input|tableinput)\{([^}]+)\}', load, text)

main = expand(src/'main.tex')
supp = expand(src/'supplementary.tex')
bib = (src/'references.bib').read_text()
keys = re.findall(r'@\w+\s*\{([^,]+),', bib)
assert len(keys) == len(set(keys)) == 26
all_cites = set()
counts = {}
for stem, text, build in [('main',main,'build_main'), ('supplementary',supp,'build_supplement')]:
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    refs = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', text)
    cites = {k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',text) for k in group.split(',')}
    assert len(labels) == len(set(labels))
    assert set(refs) <= set(labels)
    assert cites <= set(keys)
    all_cites |= cites
    bbl = (v/build/(stem+'.bbl')).read_text()
    printed = set(re.findall(r'\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}', bbl))
    assert printed == cites, (stem, printed^cites)
    log = (v/build/(stem+'.log')).read_text(errors='replace')
    for bad in ['Overfull', 'undefined', 'multiply defined', 'Missing character', 'ignored error']:
        assert bad not in log, (stem,bad)
    words = run(['texcount','-inc','-sum',stem+'.tex'],src,'texcount_'+stem+'.txt')
    assert 'not found' not in words and '(errors:' not in words
    info = run(['pdfinfo',str(v/build/(stem+'.pdf'))],src,stem+'_pdfinfo.txt')
    fonts = run(['pdffonts',str(v/build/(stem+'.pdf'))],src,stem+'_fonts.txt')
    for line in fonts.splitlines()[2:]:
        if line.strip():
            assert re.search(r'\byes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$',line), line
            assert 'Type 3' not in line
    counts[stem] = {
        'pages': int(re.search(r'Pages:\s+(\d+)',info)[1]),
        'texcount_text_words': int(re.findall(r'Words in text:\s+(\d+)',words)[-1]),
        'texcount_sum': int(re.findall(r'Sum count:\s+(\d+)',words)[-1]),
        'citations_printed':len(cites), 'labels':len(labels),
        'overfull_boxes':0, 'undefined_refs_citations':0,
        'fonts':'all embedded; no Type 3',
        'underfull_boxes':log.count('Underfull'),
    }
assert all_cites == set(keys), set(keys)-all_cites
abstract = main.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
abstract_ws = len(abstract.split())
abstract_tokens = len(re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?",abstract))
assert abstract_ws < 250 and abstract_tokens < 250
highlights = (src/'highlights.txt').read_text().splitlines()
assert 3 <= len(highlights) <= 5 and max(map(len, highlights)) <= 85
assert main.count('\\sep')+1 == 6
assert len(re.findall(r'\\section\{', main)) == 7
assert main.rfind('\\section*{Acknowledgements}') > main.rfind('\\section*{Declaration')
# Refresh output targets only after successful local builds.
shutil.copy2(v/'build_main/main.pdf',sc/'Speech_Communication_manuscript.pdf')
shutil.copy2(v/'build_supplement/supplementary.pdf',sc/'Speech_Communication_supplementary.pdf')
files = [src/p for p in ['main.tex','supplementary.tex','references.bib','elsarticle.cls',
         'elsarticle-harv.bst','highlights.txt','README.md','AUTHOR_CONFIRM.md']]
files += sorted((src/'sections').glob('*.tex')) + sorted((src/'tables').glob('*.tex'))
files += [src/'figures/Figure_5_cost_microbenchmark.pdf',src/'assets/generate_tables.py',src/'assets/numeric_checks.json']
archive = sc/'Speech_Communication_LaTeX_source.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(files):
        item = zipfile.ZipInfo(path.relative_to(src).as_posix(), (2026,9,8,0,0,0))
        item.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(item,path.read_bytes())
extract = v/'source_zip_recompile'
assert not extract.exists(), 'Use a fresh extraction directory'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    z.extractall(extract)
for stem in ['main','supplementary']:
    run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],extract,'zip_'+stem+'_build.txt')
    log = (extract/(stem+'.log')).read_text(errors='replace')
    for bad in ['Overfull','undefined','multiply defined','ignored error']:
        assert bad not in log, (stem,bad)
    for pdf, outfile in [(v/('build_main' if stem=='main' else 'build_supplement')/(stem+'.pdf'),v/(stem+'_text.txt')),
                         (extract/(stem+'.pdf'),v/('zip_'+stem+'_text.txt'))]:
        subprocess.run(['pdftotext','-layout',str(pdf),str(outfile)],check=True)
    assert (v/(stem+'_text.txt')).read_bytes() == (v/('zip_'+stem+'_text.txt')).read_bytes()
# Render every final page, without inspecting images.
for stem, filename in [('main','Speech_Communication_manuscript.pdf'),('supplementary','Speech_Communication_supplementary.pdf')]:
    png = v/('png_'+stem)
    png.mkdir(exist_ok=True)
    subprocess.run(['pdftoppm','-png','-r','110',str(sc/filename),str(png/'page')],check=True)
    assert len(list(png.glob('page-*.png'))) == counts[stem]['pages']
    counts[stem]['png_directory'] = str(png)
report = {
    'counts':counts, 'abstract_whitespace_words':abstract_ws,
    'abstract_punctuation_split_tokens':abstract_tokens,
    'highlight_characters':list(map(len,highlights)), 'keywords':6,
    'bibliography_entries_used_across_package':len(all_cites),
    'zip_entries':len(files), 'zip_rebuild_text_identical':True,
    'visual_gate':'PENDING main orchestrator; PNGs not opened by editing agent',
    'formal_revision_schema':'NOT VALIDATED',
    'independent_scientific_review':'not performed; no delegation tool',
    'fresh_citation_existence_retraction_check':'not performed',
    'artifacts_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [archive,sc/'Speech_Communication_manuscript.pdf',sc/'Speech_Communication_supplementary.pdf']},
}
(v/'technical_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
