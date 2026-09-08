from pathlib import Path
import subprocess,zipfile,shutil,json,re,hashlib,difflib
r=Path('D:/project/my/research/streamllm_p2');sc=r/'paper2/tougao/SC';src=sc/'latex';v=r/'paper2/review/sc_revision_validation_2026-09-08';o=v/'supp_bib_repair'
with zipfile.ZipFile(o/'before.zip') as z:
 old=z.read('paper2/tougao/SC/latex/supplementary.tex').decode();new=(src/'supplementary.tex').read_text()
 expected=old.replace('\\bibliographystyle{elsarticle-harv}\n\\bibliography{references}','\\begingroup\n\\setlength{\\bibsep}{0pt}\n\\interlinepenalty=10000\n\\bibliographystyle{elsarticle-harv}\n\\bibliography{references}\n\\endgroup')
 assert expected==new
 (o/'layout.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/supplementary.tex',tofile='b/supplementary.tex')))
 (o/'old_supplement.pdf').write_bytes(z.read('paper2/tougao/SC/Speech_Communication_supplementary.pdf'))
shutil.copy2(sc/'Speech_Communication_manuscript.pdf',o/'old_main.pdf')
for stem,bdir,target in [('main','build_main','Speech_Communication_manuscript.pdf'),('supplementary','build_supplement','Speech_Communication_supplementary.pdf')]:
 log=(v/bdir/(stem+'.log')).read_text(errors='replace')
 for bad in ['Overfull','undefined','multiply defined','ignored error']:assert bad not in log,(stem,bad)
 shutil.copy2(v/bdir/(stem+'.pdf'),sc/target)
archive=sc/'Speech_Communication_LaTeX_source.zip'
with zipfile.ZipFile(archive) as z:names=z.namelist()
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for name in names:
  item=zipfile.ZipInfo(name,(2026,9,8,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED;z.writestr(item,(src/name).read_bytes())
e=o/'zip_recompile';e.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None;z.extractall(e)
counts={}
for stem,target,start,oldpdf in [('main','Speech_Communication_manuscript.pdf',25,'old_main.pdf'),('supplementary','Speech_Communication_supplementary.pdf',8,'old_supplement.pdf')]:
 p=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],cwd=e,capture_output=True);(o/(stem+'_zip_build.txt')).write_bytes(p.stdout+p.stderr);assert p.returncode==0
 log=(e/(stem+'.log')).read_text(errors='replace')
 for bad in ['Overfull','undefined','multiply defined','ignored error']:assert bad not in log,(stem,bad)
 for pdf,text in [(sc/target,o/(stem+'_text.txt')),(e/(stem+'.pdf'),o/('zip_'+stem+'_text.txt')),(o/oldpdf,o/('old_'+stem+'_text.txt'))]:subprocess.run(['pdftotext','-layout','-enc','UTF-8',str(pdf),str(text)],check=True)
 assert (o/(stem+'_text.txt')).read_bytes()==(o/('zip_'+stem+'_text.txt')).read_bytes()
 oldpages=(o/('old_'+stem+'_text.txt')).read_text().split('\f');newpages=(o/(stem+'_text.txt')).read_text().split('\f')
 assert oldpages[:start-1]==newpages[:start-1],(stem,'earlier pages changed')
 info=subprocess.check_output(['pdfinfo',str(sc/target)]).decode(errors='replace');pages=int(re.search(r'Pages:\s+(\d+)',info)[1]);(o/(stem+'_pdfinfo.txt')).write_text(info)
 fonts=subprocess.check_output(['pdffonts',str(sc/target)]).decode(errors='replace');(o/(stem+'_fonts.txt')).write_text(fonts)
 for line in fonts.splitlines()[2:]:
  if line.strip():assert re.search(r'\byes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$',line) and 'Type 3' not in line,line
 png=o/('png_'+stem);png.mkdir(exist_ok=True)
 subprocess.run(['pdftoppm','-png','-r','110','-f',str(start),'-l',str(pages),str(sc/target),str(png/'page')],check=True)
 counts[stem]={'pages':pages,'earlier_page_text_unchanged':list(range(1,start)),'rendered_pages':[start,pages],'png_directory':str(png),'png_count':len(list(png.glob('*.png'))),'overfull_boxes':0,'undefined_refs_citations':0,'fonts':'all embedded; no Type 3','zip_recompile_text_identical':True}
# Source bibliography unchanged; no reference removed or URL altered.
base_bib=subprocess.check_output(['git','-C',str(r),'show','HEAD:paper2/tougao/SC/latex/references.bib']).decode()
assert base_bib.splitlines()==(src/'references.bib').read_text().splitlines()
report={'scientific_check':'completed by orchestrator-dispatched read-only agent 6343...; all 30 R intervals and E3 PASS; no independent-error claim','supplement_scientific_content':'exactly unchanged after removing bibliography-layout wrapper','reference_database_unchanged':True,'counts':counts,'zip_entries':len(names),'visual_status':'supplement earlier targeted 5-8 PASS; latest page8 recheck pending; main25-end URL recheck pending; no images inspected','artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [archive,sc/'Speech_Communication_manuscript.pdf',sc/'Speech_Communication_supplementary.pdf']}}
(o/'checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
