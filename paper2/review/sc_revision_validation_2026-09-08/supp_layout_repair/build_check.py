from pathlib import Path
import subprocess,zipfile,shutil,re,json,hashlib,difflib
r=Path('D:/project/my/research/streamllm_p2');sc=r/'paper2/tougao/SC';src=sc/'latex';v=r/'paper2/review/sc_revision_validation_2026-09-08';o=v/'supp_layout_repair'
with zipfile.ZipFile(o/'before.zip') as z: old=z.read('paper2/tougao/SC/latex/supplementary.tex').decode()
new=(src/'supplementary.tex').read_text()
oldtable=old[old.index('\\begin{table}[p]'):old.index('\\end{table}',old.index('\\begin{table}[p]'))+len('\\end{table}')]
newtable=new[new.index('\\begingroup\\small'):new.index('\\endgroup',new.index('\\begingroup\\small'))+len('\\endgroup')]
normalized=new.replace('\\usepackage{needspace}\n','').replace('\\par\\Needspace{6\\baselineskip}\n','').replace(newtable,oldtable).replace('\\bibliographystyle{elsarticle-harv}','\\clearpage\n\\bibliographystyle{elsarticle-harv}')
assert normalized==old
assert re.search(r'\\caption\{(.*?)\}\\label',oldtable)[1]==re.search(r'\\caption\{(.*?)\}\\label',newtable)[1]
(o/'layout.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/supplementary.tex',tofile='b/supplementary.tex')))
log=(v/'build_supplement/supplementary.log').read_text(errors='replace')
for bad in ['Overfull','undefined','multiply defined','ignored error']:assert bad not in log,bad
mainsha=hashlib.sha256((sc/'Speech_Communication_manuscript.pdf').read_bytes()).hexdigest()
shutil.copy2(v/'build_supplement/supplementary.pdf',sc/'Speech_Communication_supplementary.pdf')
archive=sc/'Speech_Communication_LaTeX_source.zip'
with zipfile.ZipFile(archive) as z:names=z.namelist()
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for name in names:
  item=zipfile.ZipInfo(name,(2026,9,8,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED;z.writestr(item,(src/name).read_bytes())
e=o/'zip_recompile';e.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None;z.extractall(e)
for stem in ['main','supplementary']:
 p=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],cwd=e,capture_output=True);(o/(stem+'_zip_build.txt')).write_bytes(p.stdout+p.stderr);assert p.returncode==0
 s=(e/(stem+'.log')).read_text(errors='replace')
 for bad in ['Overfull','undefined','multiply defined','ignored error']:assert bad not in s,(stem,bad)
for pdf,dst in [(v/'build_supplement/supplementary.pdf',o/'new_text.txt'),(e/'supplementary.pdf',o/'zip_text.txt'),(sc/'Speech_Communication_supplementary.pdf',o/'current_text.txt')]:subprocess.run(['pdftotext','-layout','-enc','UTF-8',str(pdf),str(dst)],check=True)
assert (o/'new_text.txt').read_bytes()==(o/'zip_text.txt').read_bytes()==(o/'current_text.txt').read_bytes()
with zipfile.ZipFile(o/'before.zip') as z:(o/'old.pdf').write_bytes(z.read('paper2/tougao/SC/Speech_Communication_supplementary.pdf'))
subprocess.run(['pdftotext','-layout','-enc','UTF-8',str(o/'old.pdf'),str(o/'old_text.txt')],check=True)
a=(o/'old_text.txt').read_text().split('\f');b=(o/'new_text.txt').read_text().split('\f');changed=[i+1 for i,(x,y) in enumerate(zip(a,b)) if x!=y];print('Text changed pages',changed)
start=min(changed);end=len([p for p in b if p.strip()]);png=o/'png_supplementary';png.mkdir(exist_ok=True)
subprocess.run(['pdftoppm','-png','-r','110','-f',str(start),'-l',str(end),str(sc/'Speech_Communication_supplementary.pdf'),str(png/'page')],check=True)
assert hashlib.sha256((sc/'Speech_Communication_manuscript.pdf').read_bytes()).hexdigest()==mainsha
report={'scientific_review':'PASS reported by user: all 30 R CIs plus E3 accurate','layout_source_content_unchanged':True,'pages':end,'changed_text_pages':changed,'rendered_range':[start,end],'png_directory':str(png),'png_count':len(list(png.glob('*.png'))),'main_pdf_unchanged':True,'zip_entries':len(names),'both_zip_documents_compile':True,'supplement_zip_text_identical':True,'overfull':0,'undefined_refs_citations':0,'visual_status':'repair awaiting judge; main judge pending; images not inspected','artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [archive,sc/'Speech_Communication_supplementary.pdf',sc/'Speech_Communication_manuscript.pdf']}}
(o/'checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
