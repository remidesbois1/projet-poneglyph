/** Visual evidence of skipped lines; boxes over the actual cached crop. */
const fs=require('node:fs'),path=require('node:path'),R=require('./runtime.cjs');
const rows=fs.readFileSync(path.join(__dirname,'results/ablation_val_300.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
const ids=[5556,4195,4254,6171,3334,5562,5568,4427,2606,4347,2779,4351];
async function main(){
 for(let start=0;start<ids.length;start+=6){
  const items=ids.slice(start,start+6),canvas=R.createCanvas(1600,1440),ctx=canvas.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,1600,1440);
  for(let i=0;i<items.length;i++){
   const row=rows.find(r=>r.id===items[i]),img=await R.loadImage(path.join(R.CACHE,row.crop));
   for(let variant=0;variant<2;variant++){
    const ox=(i%2)*800+variant*400,oy=Math.floor(i/2)*480,boxes=variant?row.thresholds['0.1'].boxes:row.boxes;
    ctx.fillStyle='black';ctx.font='18px sans-serif';ctx.fillText(`ID ${row.id} | seuil ${variant?'.10':'.25'} | ${boxes.length} lignes`,ox+8,oy+25);
    const scale=Math.min(380/img.width,420/img.height),x=ox+10+(380-img.width*scale)/2,y=oy+44;
    ctx.drawImage(img,x,y,img.width*scale,img.height*scale);
    boxes.forEach((b,index)=>{ctx.strokeStyle=b.conf<.25?'#d03030':'#007050';ctx.lineWidth=2;ctx.strokeRect(x+b.x1*scale,y+b.y1*scale,(b.x2-b.x1)*scale,(b.y2-b.y1)*scale);ctx.fillStyle=ctx.strokeStyle;ctx.font='13px sans-serif';ctx.fillText(`${index+1}:${b.conf.toFixed(2)}`,x+b.x1*scale,y+b.y1*scale-3);});
   }
  }
  const file=path.join(__dirname,'results/visual',`missed_lines_${start+1}.png`);fs.writeFileSync(file,canvas.toBuffer('image/png'));console.log(file);
 }
}
main().catch(e=>{console.error(e);process.exitCode=1});
