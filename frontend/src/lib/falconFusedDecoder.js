// Single-token Falcon decoder. ONNX Runtime still performs the image/prompt prefill.
// We reuse its exact mixed-FP16/FP32 weights and keep a fixed-size KV cache on GPU.
const PREFIX = 'enable f16;\n';
const bind = (i, name, type = 'f16', write = false) =>
    `@group(0) @binding(${i}) var<storage, ${write ? 'read_write' : 'read'}> ${name}: array<${type}>;\n`;
const reduce = `var<workgroup> sums: array<f32, 256>;
fn sum(value: f32, lane: u32, width: u32) -> f32 {
    sums[lane] = value; workgroupBarrier();
    var stride = width / 2u;
    loop { if (stride == 0u) { break; }
        if (lane < stride) { sums[lane] += sums[lane + stride]; }
        workgroupBarrier(); stride /= 2u;
    }
    return sums[0];
}`;

export class FalconFusedDecoder {
    constructor(device, options = {}) {
        this.device = device; this.options = options; this.weights = {};
        this.buffers = []; this.pipelines = new Map();
        this.tasks = []; this.sequenceBuffers = []; this.generatedCount = 0; this.readbacks = 0;
        device.lost.then(info => { this.deviceLost = info.message || 'Périphérique WebGPU perdu.'; });
    }
    buffer(bytes, label, data) {
        const buffer = this.device.createBuffer({ label, size: Math.max(4, Math.ceil(bytes / 4) * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: !!data });
        if (data) {
            new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
            buffer.unmap();
        }
        this.buffers.push(buffer); return buffer;
    }
    kernel(label, code, buffers, groups) {
        let pipeline = this.pipelines.get(code);
        if (!pipeline) {
            const shader = this.device.createShaderModule({ label, code: PREFIX + code });
            pipeline = this.device.createComputePipeline({ label, layout: 'auto', compute: { module: shader, entryPoint: 'main' } });
            this.pipelines.set(code, pipeline);
        }
        const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
            entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
        return { label, pipeline, group, groups: Array.isArray(groups) ? groups : [groups] };
    }
    dispatch(pass, task) {
        pass.setPipeline(task.pipeline); pass.setBindGroup(0, task.group); pass.dispatchWorkgroups(...task.groups);
    }
    async load(index, data) {
        const temp = []; const encoder = this.device.createCommandEncoder(); const pass = encoder.beginComputePass();
        this.device.pushErrorScope('out-of-memory'); this.device.pushErrorScope('validation');
        let failure;
        try {
        for (const [name, info] of Object.entries(index.weights)) {
            const bytes = info.bits ? Uint16Array.from(info.bits) : data.subarray(info.offset, info.offset + info.bytes);
            const uploaded = this.buffer(bytes.byteLength, name, bytes);
            if (!info.transpose) { this.weights[name] = uploaded; continue; }
            const [k, n] = info.shape, type = info.dtype === 'float32' ? 'f32' : 'f16';
            const dst = this.buffer(info.bytes, `${name}.rows`);
            const code = bind(0, 'src', type) + bind(1, 'dst', type, true) + `
                @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>) {
                    let i=id.x+id.y*16776960u; if(i>=${k * n}u){return;}
                    dst[i]=src[(i%${k}u)*${n}u+i/${k}u];
                }`;
            const groups = Math.ceil(k * n / 256);
            this.dispatch(pass, this.kernel('transpose Falcon weights', code, [uploaded, dst], [Math.min(groups, 65535), Math.ceil(groups / 65535)]));
            this.weights[name] = dst; temp.push(uploaded);
        }
        pass.end(); this.device.queue.submit([encoder.finish()]); await this.device.queue.onSubmittedWorkDone();
        temp.forEach(buffer => buffer.destroy()); this.buffers = this.buffers.filter(buffer => !temp.includes(buffer));
        this.weightBytes = this.buffers.reduce((sum, buffer) => sum + buffer.size, 0);
        } catch (error) { failure = error; }
        const validation = await this.device.popErrorScope(); const memory = await this.device.popErrorScope();
        if (failure) throw failure;
        if (validation || memory) throw new Error(`Poids Falcon WebGPU : ${(validation || memory).message}`);
    }
    async begin(cache, length, position, token, maxTokens) {
        this.end(); this.device.pushErrorScope('out-of-memory'); this.device.pushErrorScope('validation');
        let failure;
        try {
        const first = this.buffers.length; this.sequenceStart = first;
        const b = (n, bytes = 2, label = 'Falcon activation') => this.buffer(n * bytes, label);
        const tasks = []; const add = (...args) => tasks.push(this.kernel(...args));
        const weight = name => this.weights[name];
        const capacity = Math.ceil((length + maxTokens + 8) / 256) * 256;
        this.params = this.buffer(16, 'Falcon token-position-length', new Uint32Array([token, position, length, 0]));
        this.tokens = b(maxTokens + 8, 4, 'Falcon output tokens');
        this.readback = this.device.createBuffer({ size: 16 + (maxTokens + 8) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const angles = new Float32Array((maxTokens + 8) * 64);
        for (let i = 0; i < maxTokens + 8; i++) for (let f = 0; f < 32; f++) {
            const angle = f < 16 ? Math.fround((position + i) / (10000 ** (2 * f / 32))) : 0;
            angles[i * 64 + f] = Math.cos(angle); angles[i * 64 + 32 + f] = Math.sin(angle);
        }
        const rotary = this.buffer(angles.byteLength, 'Falcon rotary', angles);
        const x = b(768, 4, 'Falcon residual'), norm = b(768), qkv = b(2048), query = b(1024);
        const scores = b(16 * capacity, 4), probabilities = b(16 * capacity);
        const valueParts = b(16 * 8 * 64, 4), core = b(1024), attended = b(768);
        const projection = b(4608), ff = b(2304, 4), down = b(768, 4);
        const encoder = this.device.createCommandEncoder();

        const gemv = (src, name, dst, k, n, inputType = 'f16', weightType = 'f16', outputType = 'f16') => {
            const cast = outputType === 'f16' ? 'f16(value)' : 'value';
            if (this.device.features.has('subgroups') && this.options.gemv !== 'shared') {
                const code = 'enable subgroups;\n' + bind(0, 'x', inputType) + bind(1, 'w', weightType) + bind(2, 'y', outputType, true) + `
                    var<workgroup> parts:array<f32,128>;
                    @compute @workgroup_size(128) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>,
                        @builtin(subgroup_size) size:u32,@builtin(subgroup_invocation_id) slane:u32){
                        let row=g.x*4u;var a=0.0;var b=0.0;var c=0.0;var d=0.0;
                        for(var j=lane;j<${k}u;j+=128u){let v=f32(x[j]);
                            a+=v*f32(w[row*${k}u+j]);b+=v*f32(w[(row+1u)*${k}u+j]);
                            c+=v*f32(w[(row+2u)*${k}u+j]);d+=v*f32(w[(row+3u)*${k}u+j]);}
                        let sa=subgroupAdd(a);let sb=subgroupAdd(b);let sc=subgroupAdd(c);let sd=subgroupAdd(d);
                        if(slane==0u){let p=(lane/size)*4u;parts[p]=sa;parts[p+1u]=sb;parts[p+2u]=sc;parts[p+3u]=sd;}
                        workgroupBarrier();
                        if(lane<4u&&row+lane<${n}u){var value=0.0;for(var i=0u;i<128u/size;i++){value+=parts[i*4u+lane];}y[row+lane]=${cast};}
                    }`;
                add(`Falcon gemv ${name}`, code, [src, weight(name), dst], Math.ceil(n / 4)); return;
            }
            const width = Math.min(n, 32768);
            const code = bind(0, 'x', inputType) + bind(1, 'w', weightType) + bind(2, 'y', outputType, true) + reduce + `
                @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
                    let row=g.x+g.y*${width}u;if(row>=${n}u){return;}var acc=0.0;
                    for(var j=lane;j<${k}u;j+=64u){acc+=f32(x[j])*f32(w[row*${k}u+j]);}
                    let value=sum(acc,lane,64u);if(lane==0u){y[row]=${cast};}
                }`;
            add(`Falcon gemv ${name}`, code, [src, weight(name), dst], [width, Math.ceil(n / width)]);
        };
        const mlp = (src, name) => {
            if (!this.device.features.has('subgroups') || this.options.fusedMlp === false) {
                gemv(src, name, projection, 768, 4608);
                add('Falcon ReLU squared gate', bind(0, 'projection') + bind(1, 'ff', 'f32', true) + `
                    @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){
                        let i=id.x;let gate=max(f32(projection[i*2u]),0.0);ff[i]=gate*gate*f32(projection[i*2u+1u]);
                    }`, [projection, ff], 9);
                return;
            }
            const code = 'enable subgroups;\n' + bind(0, 'x') + bind(1, 'w') + bind(2, 'ff', 'f32', true) + `
                var<workgroup> parts:array<f32,128>;
                @compute @workgroup_size(128) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>,
                    @builtin(subgroup_size) size:u32,@builtin(subgroup_invocation_id) slane:u32){
                    let row=g.x*4u;var a=0.0;var b=0.0;var c=0.0;var d=0.0;
                    for(var j=lane;j<768u;j+=128u){let v=f32(x[j]);
                        a+=v*f32(w[row*768u+j]);b+=v*f32(w[(row+1u)*768u+j]);
                        c+=v*f32(w[(row+2u)*768u+j]);d+=v*f32(w[(row+3u)*768u+j]);}
                    let sa=subgroupAdd(a);let sb=subgroupAdd(b);let sc=subgroupAdd(c);let sd=subgroupAdd(d);
                    if(slane==0u){let p=(lane/size)*4u;parts[p]=sa;parts[p+1u]=sb;parts[p+2u]=sc;parts[p+3u]=sd;}
                    workgroupBarrier();
                    if(lane<2u){var gate=0.0;var up=0.0;
                        for(var i=0u;i<128u/size;i++){gate+=parts[i*4u+lane*2u];up+=parts[i*4u+lane*2u+1u];}
                        let value=max(f32(f16(gate)),0.0);ff[g.x*2u+lane]=value*value*f32(f16(up));}
                }`;
            add('Falcon fused MLP projection', code, [src, weight(name), ff], 1152);
        };
        const rms = (residual, residualType = 'f16', final = false) => {
            const code = bind(0, 'x', 'f32', true) + bind(1, 'y', 'f16', true) +
                (residual ? bind(2, 'residual', residualType) : '') + (final ? bind(residual ? 3 : 2, 'weight') : '') + reduce + `
                @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane:u32){
                    var acc=0.0;for(var i=lane;i<768u;i+=256u){${residual ? 'x[i]+=f32(residual[i]);' : ''}acc+=x[i]*x[i];}
                    let denominator=sqrt(sum(acc,lane,256u)/768.0+${final ? '0.00001' : '0.0078125'});
                    for(var i=lane;i<768u;i+=256u){${final ? 'y[i]=f16(x[i]/denominator)*weight[i];' : 'y[i]=f16(x[i]/denominator);'}}
                }`;
            add('Falcon residual RMS', code, [x, norm, ...(residual ? [residual] : []), ...(final ? [weight('norm.weight')] : [])], 1);
        };
        add('Falcon embedding', bind(0, 'embedding') + bind(1, 'params', 'u32') + bind(2, 'x', 'f32', true) + `
            @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){x[id.x]=f32(embedding[params[0]*768u+id.x]);}
        `, [weight('tok_embeddings.weight'), this.params, x], 3);

        for (let layer = 0; layer < 22; layer++) {
            const p = `layers.${layer}.`;
            // ONNX repeats each KV head twice for GQA. Retain only one copy.
            const keys = b(8 * capacity * 64, 2, `Falcon keys ${layer}`);
            const values = b(8 * capacity * 64, 2, `Falcon values ${layer}`);
            for (let head = 0; head < 8; head++) for (const [kind, dest] of [[0, keys], [1, values]]) {
                const offset = ((layer * 2 + kind) * 16 + head * 2) * length * 64 * 2;
                encoder.copyBufferToBuffer(cache.gpuBuffer, offset, dest, head * capacity * 64 * 2, length * 64 * 2);
            }
            rms(layer ? down : null, 'f32');
            gemv(norm, p + 'attention.wqkv.weight', qkv, 768, 2048);
            add('Falcon QK RMS rotary and cache', bind(0, 'qkv') + bind(1, 'rotary', 'f32') + bind(2, 'params', 'u32') +
                bind(3, 'query', 'f16', true) + bind(4, 'keys', 'f16', true) + bind(5, 'values', 'f16', true) + `
                var<workgroup> qs:array<f32,64>;var<workgroup> ks:array<f32,64>;
                var<workgroup> qn:array<f32,64>;var<workgroup> kn:array<f32,64>;
                @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
                    let h=g.x;let q=f32(qkv[h*64u+lane]);let k=f32(qkv[1024u+(h/2u)*64u+lane]);
                    qs[lane]=q*q;ks[lane]=k*k;workgroupBarrier();
                    for(var s=32u;s>0u;s/=2u){if(lane<s){qs[lane]+=qs[lane+s];ks[lane]+=ks[lane+s];}workgroupBarrier();}
                    qn[lane]=f32(f16(q/sqrt(qs[0]/64.0+0.0078125)));
                    kn[lane]=f32(f16(k/sqrt(ks[0]/64.0+0.0078125)));workgroupBarrier();
                    let pair=lane/2u;let a=rotary[params[3]*64u+pair];let b=rotary[params[3]*64u+32u+pair];
                    let even=pair*2u;var qr=qn[even]*a-qn[even+1u]*b;var kr=kn[even]*a-kn[even+1u]*b;
                    if((lane%2u)==1u){qr=qn[even]*b+qn[even+1u]*a;kr=kn[even]*b+kn[even+1u]*a;}
                    query[h*64u+lane]=f16(qr);
                    if((h%2u)==0u){let i=((h/2u)*${capacity}u+params[2])*64u+lane;
                        keys[i]=f16(kr);values[i]=qkv[1536u+(h/2u)*64u+lane];}
                }`, [qkv, rotary, this.params, query, keys, values], 16);
            add('Falcon attention scores', bind(0, 'q') + bind(1, 'keys') + bind(2, 'params', 'u32') + bind(3, 'scores', 'f32', true) + reduce + `
                @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
                    let pos=g.x;let h=g.y;if(pos>params[2]){return;}
                    let result=f16(sum(f32(q[h*64u+lane])*f32(keys[((h/2u)*${capacity}u+pos)*64u+lane]),lane,64u));
                    if(lane==0u){scores[h*${capacity}u+pos]=f32(result*f16(0.125));}
                }`, [query, keys, this.params, scores], [capacity, 16]);
            add('Falcon attention softmax with sink', bind(0, 'scores', 'f32') + bind(1, 'params', 'u32') + bind(2, 'sinks') + bind(3, 'prob', 'f16', true) + reduce + `
                @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
                    let base=g.x*${capacity}u;var maximum=f32(sinks[g.x]);
                    for(var j=lane;j<=params[2];j+=256u){maximum=max(maximum,scores[base+j]);}
                    sums[lane]=maximum;workgroupBarrier();
                    for(var s=128u;s>0u;s/=2u){if(lane<s){sums[lane]=max(sums[lane],sums[lane+s]);}workgroupBarrier();}
                    maximum=sums[0];workgroupBarrier();var acc=select(0.0,exp(f32(sinks[g.x])-maximum),lane==0u);
                    for(var j=lane;j<=params[2];j+=256u){acc+=exp(scores[base+j]-maximum);}
                    let denominator=sum(acc,lane,256u);
                    for(var j=lane;j<=params[2];j+=256u){prob[base+j]=f16(exp(scores[base+j]-maximum)/denominator);}
                }`, [scores, this.params, weight(p + 'attention.sinks'), probabilities], 16);
            add('Falcon attention value partial', bind(0, 'prob') + bind(1, 'values') + bind(2, 'params', 'u32') + bind(3, 'parts', 'f32', true) + `
                @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
                    let h=g.x;let count=(params[2]+8u)/8u;let end=min((g.y+1u)*count,params[2]+1u);var acc=0.0;
                    for(var j=g.y*count;j<end;j++){acc+=f32(prob[h*${capacity}u+j])*f32(values[((h/2u)*${capacity}u+j)*64u+lane]);}
                    parts[(h*8u+g.y)*64u+lane]=acc;
                }`, [probabilities, values, this.params, valueParts], [16, 8]);
            add('Falcon attention value sum', bind(0, 'parts', 'f32') + bind(1, 'out', 'f16', true) + `
                @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
                    var acc=0.0;for(var i=0u;i<8u;i++){acc+=parts[(g.x*8u+i)*64u+lane];}out[g.x*64u+lane]=f16(acc);
                }`, [valueParts, core], 16);
            gemv(core, p + 'attention.wo.weight', attended, 1024, 768);
            rms(attended);
            mlp(norm, p + 'feed_forward.w13.weight');
            gemv(ff, p + 'feed_forward.w2.weight', down, 2304, 768, 'f32', 'f32', 'f32');
        }
        rms(down, 'f32', true);
        const logits = b(65536), maxima = b(256, 4), indices = b(256, 4);
        gemv(norm, 'output.weight', logits, 768, 65536);
        const argmax = (last = false) => bind(0, 'x', last ? 'f32' : 'f16') + bind(1, 'values', last ? 'u32' : 'f32', true) +
            bind(2, 'indices', 'u32', true) + (last ? bind(3, 'inputIds', 'u32') : '') + `
            var<workgroup> m:array<f32,256>;var<workgroup> ids:array<u32,256>;
            @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
                let i=g.x*256u+lane;var v=-3.4e38;var id=4294967295u;
                ${last ? 'let candidate=x[lane];let candidateId=inputIds[lane];v=candidate;id=candidateId;' : 'v=f32(x[i]);id=i;'}
                m[lane]=v;ids[lane]=id;workgroupBarrier();
                for(var s=128u;s>0u;s/=2u){if(lane<s){if(m[lane+s]>m[lane]||(m[lane+s]==m[lane]&&ids[lane+s]<ids[lane])){m[lane]=m[lane+s];ids[lane]=ids[lane+s];}}workgroupBarrier();}
                if(lane==0u){${last ? 'indices[values[3]]=ids[0];values[0]=ids[0];values[1]++;values[2]++;values[3]++;' : 'values[g.x]=m[0];indices[g.x]=ids[0];'}}
            }`;
        add('Falcon argmax blocks', argmax(), [logits, maxima, indices], 256);
        add('Falcon argmax token', argmax(true), [maxima, this.params, this.tokens, indices], 1);
        this.tasks = tasks; this.sequenceBuffers = this.buffers.slice(first); this.generatedCount = 0; this.readbacks = 0;
        this.device.queue.submit([encoder.finish()]);
        } catch (error) { failure = error; }
        const validation = await this.device.popErrorScope(); const memory = await this.device.popErrorScope();
        if (failure) throw failure;
        if (validation || memory) throw new Error(`Décodeur Falcon WebGPU : ${(validation || memory).message}`);
    }
    async step(count = 1) {
        if (this.deviceLost) throw new Error(this.deviceLost);
        if (!Number.isInteger(count) || count < 1 || count > 8) throw new Error('Taille de soumission GPU invalide.');
        const encoder = this.device.createCommandEncoder(), pass = encoder.beginComputePass();
        for (let i = 0; i < count; i++) for (const task of this.tasks) this.dispatch(pass, task);
        pass.end(); encoder.copyBufferToBuffer(this.params, 0, this.readback, 0, 16);
        encoder.copyBufferToBuffer(this.tokens, this.generatedCount * 4, this.readback, 16, count * 4);
        this.device.queue.submit([encoder.finish()]);
        await this.readback.mapAsync(GPUMapMode.READ, 0, 16 + count * 4);
        const next = Array.from(new Uint32Array(this.readback.getMappedRange(0, 16 + count * 4)).slice(4));
        this.readback.unmap(); this.generatedCount += count; this.readbacks++; return next;
    }
    end() {
        const discard = new Set(this.sequenceStart === undefined ? [] : this.buffers.slice(this.sequenceStart));
        discard.forEach(buffer => buffer.destroy()); this.buffers = this.buffers.filter(buffer => !discard.has(buffer));
        this.sequenceStart = undefined; this.readback?.destroy(); this.readback = null; this.sequenceBuffers = []; this.tasks = [];
        this.generatedCount = 0; this.readbacks = 0;
    }
    stats() { return { kernelsPerStep: this.tasks.length, computedSteps: this.generatedCount, readbacks: this.readbacks,
        weightBytes: this.weightBytes, sequenceBytes: (this.sequenceBuffers || []).reduce((sum, buffer) => sum + buffer.size, 0),
        subgroups: this.device.features.has('subgroups') && this.options.gemv !== 'shared' }; }
    dispose() { this.end(); this.buffers.forEach(buffer => buffer.destroy()); this.buffers = []; this.weights = {}; }
}
