#!/usr/bin/env python3
"""Build explicitly blocked-replay diagnostic derivatives; no model or inference."""
from __future__ import annotations
import argparse,csv,hashlib,html,io,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pypdf import PdfReader,PdfWriter

STEMS=['native_location_comparison','parameter_randomization']
BANNER='DIAGNOSTIC ONLY — probability replay check failed (2/100; classes unchanged).'
DETAIL='Tolerance 0.0001 unchanged. Not anatomical or clinical validation. See replay limitation.'
EXPECTED_FAILURES={'image_005':0.00011771917343139648,'image_010':0.00037366151809692383}


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def safe(root,name):
    p=Path(name);require(not p.is_absolute() and '..' not in p.parts and not (root/p).is_symlink() and (root/p).resolve().is_relative_to(root.resolve()),'Unsafe artifact path');return root/p


def inventory(root):
    result={}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),'Symlinks are not accepted')
        if p.is_file():result[p.relative_to(root).as_posix()]=sha(p)
        else:require(p.is_dir(),'Nonregular artifact')
    return result


def verify_inventory(root,marker_name,status):
    marker=read(root/marker_name);expected=read(root/'SHA256SUMS.json');require(marker['status']==status and marker['manifest_sha256']==sha(root/'SHA256SUMS.json') and marker['payloads']==len(expected),'Incomplete or unexpected source status')
    for name,digest in expected.items():require(sha(safe(root,name))==digest,'Source payload changed')
    require(set(inventory(root))==set(expected)|{'SHA256SUMS.json',marker_name},'Unexpected source artifacts')
    return marker,expected


def warning_png(source,target):
    import matplotlib.font_manager as fm
    fontfile=fm.findfont('DejaVu Sans');original=Image.open(source).convert('RGB');size=max(13,min(26,original.width//66));band=max(80,int(size*3.7));canvas=Image.new('RGB',(original.width,original.height+band),'#fff0bf');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(fontfile,size);small=ImageFont.truetype(fontfile,max(10,int(size*.72)))
    draw.text((12,10),BANNER,font=font,fill='#702b00');draw.text((12,15+size),DETAIL,font=small,fill='#702b00');canvas.paste(original,(0,band));canvas.save(target)
    with Image.open(target) as copied:require(np.array_equal(np.asarray(copied)[band:],np.asarray(original)),'Warning altered original figure pixels')
    return band


def warning_pdf(source,target):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original=PdfReader(source);require(len(original.pages)==1,'Summary must be one page');page=original.pages[0];width=float(page.mediabox.width);height=float(page.mediabox.height);band=48
    fig=plt.figure(figsize=(width/72,band/72),facecolor='#fff0bf');fig.text(.015,.65,BANNER,fontsize=9,color='#702b00');fig.text(.015,.28,DETAIL,fontsize=8,color='#702b00');buf=io.BytesIO();fig.savefig(buf,format='pdf',facecolor='#fff0bf');plt.close(fig);buf.seek(0)
    writer=PdfWriter();dest=writer.add_blank_page(width=width,height=height+band);dest.merge_page(page);dest.merge_translated_page(PdfReader(buf).pages[0],0,height);writer.write(target)
    result=PdfReader(target);require(len(result.pages)==1 and 'DIAGNOSTIC ONLY' in result.pages[0].extract_text(),'Diagnostic PDF banner missing')
    return band


def appendix(source,output):
    source=source.resolve();output=output.resolve();require(source.is_dir() and not output.exists() and not output.is_relative_to(source),'New separate appendix output required')
    marker,payloads=verify_inventory(source,'attribution_complete.json','blocked_cloud_probability_replay_review');v=read(source/'verification.json');failure={r['case_id']:r for r in v['cloud_probability_or_argmax_failures']}
    require(v['case_count']==115 and v['valid_maps']==115 and v['randomization_records']==72 and v['cloud_probability_tolerance']==1e-4,'Unexpected diagnostic cohort/threshold')
    require(set(failure)==set(EXPECTED_FAILURES) and all(r['argmax_agreement'] is True and abs(r['max_probability_difference']-EXPECTED_FAILURES[k])<1e-14 for k,r in failure.items()),'This banner is specific to the recorded two same-class failures')
    panels=sorted(name for name in payloads if name.startswith('cases/') and name.endswith('.png'));require(len(panels)==115,'Expected115 panels')
    for stem in STEMS:
        for ext in ['png','pdf']:require(stem+'.'+ext in payloads,'Missing diagnostic summary')
    before=inventory(source);output.mkdir(parents=True);changed=set(panels)|{stem+'.'+ext for stem in STEMS for ext in ['png','pdf']}|{'gallery.html'};mapping={}
    for name,digest in payloads.items():
        target=safe(output,name);target.parent.mkdir(parents=True,exist_ok=True)
        if name in changed:continue
        with safe(source,name).open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
        require(sha(target)==digest,'Copied source changed')
    bands={}
    for name in panels+[stem+'.png' for stem in STEMS]:bands[name]=warning_png(source/name,output/name)
    for stem in STEMS:bands[stem+'.pdf']=warning_pdf(source/(stem+'.pdf'),output/(stem+'.pdf'))
    text='<meta charset="utf-8"><title>Diagnostic attribution — replay check failed</title><h1>'+html.escape(BANNER)+'</h1><p>'+html.escape(DETAIL)+'</p><p>The original blocked marker is preserved. These single-image explanations do not meet the frozen evaluation probability-replay tolerance. All original panel pixels below the warning band and all native arrays remain unchanged.</p><p><a href="ATTRIBUTION_REPLAY_LIMITATION.md">Full limitation</a> | <a href="source_attribution_complete.json">Original blocked status</a></p>'
    for name in panels:text+='<h2>'+html.escape(Path(name).stem)+'</h2><img width="1680" src="'+html.escape(name)+'" alt="Diagnostic Grad-CAM panel; replay check failed">'
    (output/'gallery.html').write_text(text)
    for old,new in [('attribution_complete.json','source_attribution_complete.json'),('SHA256SUMS.json','source_SHA256SUMS.json')]:shutil.copyfile(source/old,output/new);require(sha(source/old)==sha(output/new),'Original receipt changed')
    note=Path(__file__).with_name('ATTRIBUTION_REPLAY_LIMITATION.md');shutil.copyfile(note,output/note.name)
    for name,digest in payloads.items():mapping[name]=dict(source_sha256=digest,output_sha256=sha(output/name),warning_derivative=name in changed)
    provenance=dict(status='diagnostic_derivatives_original_replay_still_blocked',source_directory=source.name,source_marker_sha256=sha(source/'attribution_complete.json'),original_marker_sha256=sha(source/'attribution_complete.json'),warning_visuals_sha256={name:sha(output/name) for name in sorted(changed-{'gallery.html'})},source_inventory_sha256=sha(source/'SHA256SUMS.json'),generator_sha256=sha(Path(__file__)),limitation_note_sha256=sha(note),source_output_sha256=mapping,warning_band_height=bands,original_native_arrays_and_tables_unchanged=True,no_inference=True,no_tolerance_change=True,original_status='blocked_cloud_probability_replay_review',visual_review_required=True)
    dump(output/'diagnostic_provenance.json',provenance);require(inventory(source)==before,'Original source changed during derivative creation');inv=inventory(output);dump(output/'SHA256SUMS.json',inv);complete=dict(status='complete_diagnostic_appendix_with_blocked_replay',payloads=len(inv),manifest_sha256=sha(output/'SHA256SUMS.json'),source_marker_sha256=sha(source/'attribution_complete.json'),panel_count=115,summary_pairs=2,visual_review_required=True,no_inference=True);dump(output/'diagnostic_appendix_complete.json',complete);return complete


def atlas(report,diagnostic,output):
    report=report.resolve();diagnostic=diagnostic.resolve();output=output.resolve();require(not output.exists(),'Atlas destination must be new')
    verify_inventory(report,'report_complete.json','complete_numerically_verified_visual_review_required');verify_inventory(diagnostic,'diagnostic_appendix_complete.json','complete_diagnostic_appendix_with_blocked_replay')
    writer=PdfWriter();rows=[];page=0
    for label,source,n in [('Quantitative and robustness',report,28),('Diagnostic attribution — original replay check failed',diagnostic,2)]:
        files=sorted(source.glob('*.pdf'));require(len(files)==n,'Unexpected graph count');parent=writer.add_outline_item(label,page)
        for f in files:
            pdf=PdfReader(f);require(len(pdf.pages)==1,'One page per graph required');writer.add_page(pdf.pages[0]);title=f.stem.replace('_',' ').capitalize();writer.add_outline_item(title,page,parent=parent);page+=1;rows.append(dict(page=page,group=label,title=title,source_directory=source.name,source_file=f.name,source_sha256=sha(f),attribution_replay_status='blocked_cloud_probability_replay_review' if source==diagnostic else 'not_applicable'))
    output.mkdir(parents=True);target=output/'Round2_Diagnostic_Graphs_Atlas.pdf';writer.write(target);require(len(PdfReader(target).pages)==30,'Wrong atlas page count')
    with (output/'atlas_index.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'README.md').write_text('# Round 2 diagnostic graph atlas\n\n[All 30 pages](Round2_Diagnostic_Graphs_Atlas.pdf): 28 quantitative/robustness figures plus two visibly flagged attribution diagnostics. The attribution probability replay check remains FAILED for 2/100 cases; classes agree and the original tolerance is unchanged. This atlas does not certify attribution replay success.\n\n[All 115 warning-bearing diagnostic panels](../'+diagnostic.name+'/gallery.html). [Page index](atlas_index.csv).\n')
    value=dict(status='complete_diagnostic_atlas_with_blocked_replay',pages=30,quantitative_pages=28,diagnostic_pages=2,bookmarks=32,atlas_sha256=sha(target),attribution_replay_status='blocked_cloud_probability_replay_review',diagnostic_manifest_sha256=sha(diagnostic/'SHA256SUMS.json'),report_manifest_sha256=sha(report/'SHA256SUMS.json'),source_figures=rows,visual_review_required=True,no_inference=True);dump(output/'diagnostic_atlas_verification.json',value);return value


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='mode',required=True);x=sub.add_parser('appendix');x.add_argument('--source',type=Path,required=True);x.add_argument('--output',type=Path,required=True);x=sub.add_parser('atlas');x.add_argument('--report',type=Path,required=True);x.add_argument('--diagnostic',type=Path,required=True);x.add_argument('--output',type=Path,required=True);args=ap.parse_args();result=appendix(args.source,args.output) if args.mode=='appendix' else atlas(args.report,args.diagnostic,args.output);print(json.dumps({k:v for k,v in result.items() if k!='source_figures'}))

if __name__=='__main__':main()
