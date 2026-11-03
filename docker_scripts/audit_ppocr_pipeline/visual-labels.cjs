/** Visual labels, not inferred from transcript length or OCR errors.
 * First 100 deterministic examples of each split were inspected on contact sheets.
 * "ordinary" is horizontal PP-OCR-compatible typography, including punctuation,
 * bold dialogue, hand lettering and short expressive sounds.
 * Special layout distinguishes vertical spoken cries from non-dialogue SFX.
 * Ambiguous cases are excluded from binary router evaluation.
 */
const fs=require('node:fs'),path=require('node:path'),R=require('./runtime.cjs');
const data=JSON.parse(fs.readFileSync(path.join(R.CACHE,'dataset.json')));
const exceptions={
 val:{8553:['stylized_sfx','Curved PATAM; readable by the recognizer despite stylization.'],2721:['stylized_sfx','Vertical/curved ROAAR.'],2762:['stylized_sfx','Vertical/angled RRR.'],5343:['stylized_sfx','Angled VIOUU over drawing.'],1454:['special_layout','Vertically stacked spoken cry YAAHHH.'],7254:['special_layout','Vertically stacked spoken cry AHH.'],501:['ambiguous','Horizontal BLP: expressive sound, OCR-compatible geometry.']},
 test:{2531:['stylized_sfx','Vertical/angled SCHLOP.'],5753:['stylized_sfx','Curved FRSHH.'],6516:['stylized_sfx','CRAAC over drawing.'],6676:['stylized_sfx','Angled/vertical KLING.'],5755:['stylized_sfx','Angled/vertical KWIT.'],7045:['special_layout','Vertically stacked spoken scream AAHH!!; not necessarily SFX.'],6628:['ambiguous','Horizontal TSS: expressive sound, OCR-compatible geometry.']}
};
const labels=['val','test'].flatMap(split=>R.selectExamples(data,split,100).map(e=>({id:e.id,split,label:exceptions[split][e.id]?.[0]||'ordinary',note:exceptions[split][e.id]?.[1]||'Horizontal dialogue or ordinary punctuation; visually inspected.'})));
const out={protocol:'Direct contact-sheet review by the assistant; one annotator; exploratory labels require human confirmation. No labels derived from string length.',labels};
fs.writeFileSync(path.join(__dirname,'visual-labels.json'),JSON.stringify(out,null,2));console.log(JSON.stringify(Object.fromEntries(['val','test'].map(split=>[split,labels.filter(r=>r.split===split).reduce((a,r)=>(a[r.label]=(a[r.label]||0)+1,a),{})]))));
