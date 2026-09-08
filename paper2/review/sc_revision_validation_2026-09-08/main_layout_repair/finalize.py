from pathlib import Path
import subprocess,zipfile,shutil,json,re,hashlib,difflib
r=Path('D:/project/my/research/streamllm_p2');sc=r/'paper2/tougao/SC';src=sc/'latex';v=r/'paper2/review/sc_revision_validation_2026-09-08';o=v/'main_layout_repair'
with zipfile.ZipFile(o/'before.zip') as z:
 old=z.read('paper2/tougao/SC/latex/sections/05_experiments_results.tex').decode()
 new=(src/'sections/05_experiments_results.tex').read_text()
 fixed=old.replace('\\paragraph{Design and comparator.}','\\paragraph{Design and comparator}').replace('\\paragraph{Endpoints and uncertainty.}','\\paragraph{Endpoints and uncertainty}').replace('\\paragraph{Results.}','\\paragraph{Results}').replace('width=0.82\\linewidth,height=0.39\\textheight,keepaspectratio','width=\\linewidth')
 assert fixed==new
 patches=[]
 for name in ['main.tex','sections/05_experiments_results.tex']:
  before=z.read('paper2/tougao/SC/latex/'+name).decode();after=(src/name).read_text()
  patches.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
 (o/'layout.patch').write_text(''.join(patches))
for stem,bdir,target in [('main','build_main','Speech_Communication_manuscript.pdf'),('supplementary','build_supplement','Speech_Communication_supplementary.pdf')]:
 log=(v/bdir/(stem+'.log')).read_text(errors='replace')
 for bad in ['Overfull','undefined','multiply defined','ignored error']:assert bad not in log,(stem,bad)
 shutil.copy2(v/bdir/(stem+'.pdf'),sc/target)
archive=sc/'Speech_Communication_LaTeX_source.zip'
with zipfile.ZipFile(archive) as z:names=z.namelist()
names=sorted(set(names+['assets/generate_a1_figure.py']))
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for name in names:
  item=zipfile.ZipInfo(name,(2026,9,8,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED;z.writestr(item,(src/name).read_bytes())
e=o/'zip_recompile';e.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None;z.extractall(e)
counts={}
for stem,target,start in [('main','Speech_Communication_manuscript.pdf',16),('supplementary','Speech_Communication_supplementary.pdf',5)]:
 p=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],cwd=e,capture_output=True);(o/(stem+'_zip_build.txt')).write_bytes(p.stdout+p.stderr);assert p.returncode==0
 log=(e/(stem+'.log')).read_text(errors='replace')
 for bad in ['Overfull','undefined','multiply defined','ignored error']:assert bad not in log,(stem,bad)
 for pdf,text in [(sc/target,o/(stem+'_text.txt')),(e/(stem+'.pdf'),o/('zip_'+stem+'_text.txt'))]:subprocess.run(['pdftotext','-layout','-enc','UTF-8',str(pdf),str(text)],check=True)
 assert (o/(stem+'_text.txt')).read_bytes()==(o/('zip_'+stem+'_text.txt')).read_bytes()
 info=subprocess.check_output(['pdfinfo',str(sc/target)]).decode(errors='replace');pages=int(re.search(r'Pages:\s+(\d+)',info)[1]);(o/(stem+'_pdfinfo.txt')).write_text(info)
 fonts=subprocess.check_output(['pdffonts',str(sc/target)]).decode(errors='replace');(o/(stem+'_fonts.txt')).write_text(fonts)
 for line in fonts.splitlines()[2:]:
  if line.strip():assert re.search(r'\byes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$',line) and 'Type 3' not in line,line
 png=o/('png_'+stem);png.mkdir(exist_ok=True)
 subprocess.run(['pdftoppm','-png','-r','110','-f',str(start),'-l',str(pages),str(sc/target),str(png/'page')],check=True)
 counts[stem]={'pages':pages,'rendered_pages':[start,pages],'png_directory':str(png),'png_count':len(list(png.glob('*.png'))),'overfull_boxes':0,'undefined_refs_citations':0,'fonts':'all embedded; no Type 3','zip_recompile_text_identical':True}
report={'scientific_check':'user-reported separate PASS: all 30 R intervals and E3 accurate','main_prose_changes':'only three run-in terminal periods removed; full-width A1 figure','a1_data':'same accepted six medians/quartiles and ratios; labels enlarged in SC-only vector generator','supplement_scientific_content':'unchanged; prior layout normalization assertion passed','counts':counts,'zip_entries':len(names),'visual_status':'targeted recheck pending; no images inspected','artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [archive,sc/'Speech_Communication_manuscript.pdf',sc/'Speech_Communication_supplementary.pdf']}}
(o/'checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
