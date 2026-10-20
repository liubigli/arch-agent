"""Translate existing benchmark SVG labels without changing plotted data."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'analysis_outputs' / 'benchmark_figures_english'
OUT.mkdir(parents=True, exist_ok=True)
T = {
 'Valore':'Exact', 'esatto':'value', 'Assenza':'Absence', 'Insiemi':'Sets',
 'Coerenza':'Tool', 'con i tool':'consistency', 'Conteggi':'Counts',
 'Per':'Per', 'oggetto':'object', 'Relazioni':'Relationships', 'Coordinate':'Coordinates',
 'Astensione':'Abstention', '(senza CSV)':'(without CSV)', 'Totale':'Overall',
 'campione ridotto':'small sample',
 'Accuratezza per famiglia di domande':'Accuracy by question family',
 'Accuratezza senza il layer CSV':'Accuracy without the CSV layer',
 'scena4_VAL, 60 domande, run del 17 settembre 2026':'scena4_VAL, 60 questions, run on 17 September 2026',
 'scena4_VAL, condizione graph, run del 22 settembre 2026':'scena4_VAL, graph condition, run on 22 September 2026',
 'Le famiglie con n basso sono indicative, non conclusive.':'Families with small n provide indicative, not conclusive, results.',
 'Quanto vale il layer CSV':'The contribution of the CSV layer',
 "Stessi modelli, stesse domande, con e senza le annotazioni dell'architetto":"Same models and questions, with and without the architect's annotations",
 'con CSV':'with CSV', 'senza CSV':'without CSV',
 'accuratezza sulle 42 domande la cui risposta attesa non cambia':'accuracy on the 42 questions with unchanged expected answers',
 'Le 11 domande che dipendono dal CSV sono escluse: senza CSV la risposta corretta non è il valore ma dichiararne l\'assenza,':'The 11 CSV-dependent questions are excluded: without CSV, the correct answer is to report that the value is unavailable;',
 "e confrontarle misurerebbe due compiti diversi. L'asterisco marca le differenze che superano il test di McNemar esatto.":"comparing them would assess two different tasks. An asterisk marks significant differences in the exact McNemar test.",
 "Quando il dato non c'è, il modello lo dice?":"Does the model acknowledge missing data?",
 'Condizione senza CSV: materiale, tipologia e funzione non sono caricati. La risposta corretta è dichiararlo.':'Without CSV, material, typology and function are not loaded. The correct response is to acknowledge this.',
 'domande su 11 in cui il dato richiesto non è raggiungibile':'questions out of 11 for which the requested data are unavailable',
 'dichiara il dato non disponibile':'reports data unavailable',
 'fornisce comunque un valore':'provides a value anyway',
 'nessuna risposta prodotta':'no answer produced',
 "Nessuna risposta prodotta non è un'allucinazione: è il ciclo dell'agente che si interrompe dopo la chiamata allo strumento.":"No answer produced is not a hallucination: the agent loop stops after the tool call.",
 'Le due cose sono tenute separate perché hanno cause diverse e richiedono interventi diversi.':'These outcomes are shown separately because they have different causes and require different interventions.',
 'Quali differenze tra modelli sono reali':'Which model differences are statistically significant?',
 'Test di McNemar esatto sulle 50 domande valutabili, stesse domande per ogni modello':'Exact McNemar test on 50 scorable questions, with the same questions for every model',
 'differenza reale':'significant difference', 'non distinguibili':'not significant',
 'I modelli sulla diagonale sono ordinati per accuratezza. Soglia alpha = 0.05; con 10 confronti la correzione di Bonferroni la porta a 0.0050, e delle differenze marcate':'Diagonal models are ranked by accuracy. Alpha = 0.05; Bonferroni correction for 10 comparisons lowers it to 0.0050.',
 "come reali 5 su 6 reggono anche a quella soglia. Dentro ciascun gruppo grigio la classifica non e' sostenuta dai dati.":'Five of the six marked differences remain significant at that threshold. Rankings within grey groups are not supported by the data.',
}
for n in (7,10):
 T[f'Percentuale di risposte pienamente corrette. Le {n} domande interpretative restano escluse: richiedono giudizio umano.'] = f'Percentage of fully correct answers. The {n} interpretive questions are excluded: they require human judgement.'

NS='http://www.w3.org/2000/svg'
ET.register_namespace('',NS)
ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
for src in sorted((ROOT/'docs'/'figures').glob('*.svg')):
 parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
 tree=ET.parse(src, parser=parser)
 count=0
 for parent in list(tree.iter()):
  children=list(parent)
  for i, comment in enumerate(children):
   if comment.tag is not ET.Comment: continue
   old=(comment.text or '').strip()
   if old not in T: continue
   group=children[i+1]
   uses=group.findall(f'{{{NS}}}use')
   if not uses: continue
   xs=[float(re.search(r'translate\(([-\d.]+)',u.get('transform','')).group(1)) if 'translate(' in u.get('transform','') else float(u.get('x','0')) for u in uses]
   width=max(xs)+60
   # Retain all glyph definitions because other labels may reference them.
   for u in uses: group.remove(u)
   label=T[old]
   text=ET.SubElement(group,f'{{{NS}}}text',{
    'x':str(width/2),'y':'0','transform':'scale(1 -1)',
    'text-anchor':'middle','font-family':'DejaVu Sans, Arial, sans-serif',
    'font-size':'100',
   })
   if 'Bold' in ET.tostring(group, encoding='unicode') or ('fontweight' in group.attrib):
    text.set('font-weight','bold')
   # Prevent longer translations from extending into adjacent cells or margins.
   if len(label)*52 > width:
    text.set('textLength',str(width))
    text.set('lengthAdjust','spacingAndGlyphs')
   text.text=label
   comment.text=' '+label+' '
   count+=1
 target=OUT/(src.stem+'_en.svg')
 tree.write(target,encoding='utf-8',xml_declaration=True)
 root=tree.getroot()
 w,h=[float(v) for v in root.get('viewBox').split()[2:]]
 pxw,pxh=round(w*200/72),round(h*200/72)
 svg=ET.tostring(root,encoding='unicode')
 html=f'<html><head><style>html,body{{margin:0;overflow:hidden;background:#fcfcfb}}svg{{width:{pxw}px;height:{pxh}px;display:block}}</style></head><body>{svg}</body></html>'
 target.with_suffix('.html').write_text(html,encoding='utf-8')
 print(f'{target.name}: {count} labels; {pxw}x{pxh}')
