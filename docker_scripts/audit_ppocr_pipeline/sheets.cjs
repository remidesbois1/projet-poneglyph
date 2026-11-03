/** Contact sheets for direct visual audit. Labels must come from the crop, not text length. */
const fs=require('node:fs'),path=require('node:path'),R=require('./runtime.cjs');
const data=JSON.parse(fs.readFileSync(path.join(R.CACHE,'dataset.json')));
const out=path.join(__dirname,'results','visual');fs.mkdirSync(out,{recursive:true});
const args=process.argv.slice(2),split=args[0]||'test',n=Number(args[1]||100);
async function main(){
 const examples=R.selectExamples(data,split,n);
 for(let start=0;start<examples.length;start+=25){
  const items=examples.slice(start,start+25),canvas=R.createCanvas(1500,1500),ctx=canvas.getContext('2d');ctx.fillStyle='#ddd';ctx.fillRect(0,0,1500,1500);
  for(let i=0;i<items.length;i++){
   const e=items[i],image=await R.loadImage(path.join(R.CACHE,e.crop)),x=(i%5)*300,y=Math.floor(i/5)*300;
   ctx.fillStyle='white';ctx.fillRect(x+2,y+2,296,296);ctx.fillStyle='black';ctx.font='18px sans-serif';ctx.fillText(`${start+i+1}. ID ${e.id}`,x+8,y+23);
   const scale=Math.min(282/image.width,264/image.height);ctx.drawImage(image,x+9+(282-image.width*scale)/2,y+30+(264-image.height*scale)/2,image.width*scale,image.height*scale);
  }
  const file=path.join(out,`${split}_${start+1}_${start+items.length}.png`);fs.writeFileSync(file,canvas.toBuffer('image/png'));console.log(file);
 }
 fs.writeFileSync(path.join(out,`${split}_ids.json`),JSON.stringify(examples.map(e=>e.id)));
}
main().catch(e=>{console.error(e);process.exitCode=1});
