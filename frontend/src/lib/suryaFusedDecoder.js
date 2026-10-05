// The single-token Qwen3.5 path. ONNX still performs vision and prompt prefill.
// Weights remain FP16; residuals, normalization and DeltaNet state stay FP32.
import { suryaRotary } from './suryaPreprocessing';
const PREFIX = 'enable f16;\n';
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
const bind = (i, name, type = 'f16', rw = false) => `@group(0) @binding(${i}) var<storage, ${rw ? 'read_write' : 'read'}> ${name}: array<${type}>;\n`;

export class SuryaFusedDecoder {
    constructor(device, options = {}) {
        this.device = device; this.weights = {}; this.buffers = []; this.pipelines = new Map();
        this.options = options;
        device.lost.then(info => { this.deviceLost = info.message || 'Périphérique WebGPU perdu.'; });
    }
    buffer(bytes, label, data) {
        const b = this.device.createBuffer({ label, size: Math.max(4, Math.ceil(bytes / 4) * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: !!data });
        if (data) { new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); b.unmap(); }
        this.buffers.push(b); return b;
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
        this.device.pushErrorScope('validation');
        const temporary = [];
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        for (const [name, info] of Object.entries(index.weights)) {
            if (info.values) {
                const a = new Float32Array(info.values);
                this.weights[name] = this.buffer(a.byteLength, name, a);
            } else {
                const bytes = data.subarray(info.offset, info.offset + info.bytes);
                const uploaded = this.buffer(info.bytes, name, bytes);
                if (info.transpose) {
                    const [k, n] = info.shape;
                    const b = this.buffer(info.bytes, name + '.rows');
                    const code = bind(0, 'src') + bind(1, 'dst', 'f16', true) + `
                        @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let i = id.x; if (i >= ${k * n}u) { return; }
                            dst[i] = src[(i % ${k}u) * ${n}u + i / ${k}u];
                        }`;
                    // A two-dimensional dispatch also supports the large projection matrices.
                    const tiled = code.replace('let i = id.x;', 'let i = id.x + id.y * 16776960u;');
                    const count = Math.ceil(k * n / 256);
                    this.dispatch(pass, this.kernel('transpose', tiled, [uploaded, b], [Math.min(count, 65535), Math.ceil(count / 65535)]));
                    this.weights[name] = b; temporary.push(uploaded);
                } else this.weights[name] = uploaded;
            }
        }
        pass.end(); this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        temporary.forEach(b => b.destroy()); this.buffers = this.buffers.filter(b => !temporary.includes(b));
        this.weightBytes = this.buffers.reduce((bytes, b) => bytes + b.size, 0);
        const error = await this.device.popErrorScope();
        if (error) throw new Error(`Poids WebGPU : ${error.message}`);
    }
    // Create bind groups once per image. Only the token/position counter changes during decoding.
    async begin(cache, length, position, token, maxTokens) {
        this.end(); this.device.pushErrorScope('validation');
        const before = this.buffers.length;
        this.sequenceStart = before;
        const b = (n, dtype = 2, label = 'activation') => this.buffer(n * dtype, label);
        const weights = key => this.weights[key];
        const tasks = []; const add = (...args) => tasks.push(this.kernel(...args));
        this.params = this.buffer(16, 'token-position-length', new Uint32Array([token, position, length, 0]));
        const rotations = suryaRotary(Array.from({ length: maxTokens + 8 }, (_, i) => [position + i, position + i, position + i]));
        const angles = new Float32Array((maxTokens + 8) * 128);
        for (let i = 0; i < maxTokens + 8; i++) {
            angles.set(rotations.cos.subarray(i * 64, i * 64 + 64), i * 128);
            angles.set(rotations.sin.subarray(i * 64, i * 64 + 64), i * 128 + 64);
        }
        const rotary = this.buffer(angles.byteLength, 'rotary-table', angles);
        this.tokens = b(maxTokens + 8, 4, 'generated-tokens');
        this.readback = this.device.createBuffer({ size: 16 + (maxTokens + 8) * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const x = b(1024, 4, 'residual'), norm = b(1024), attended = b(1024);
        const qkv = b(6144), z = b(2048), a = b(16), beta = b(16), mixed = b(6144);
        const qk = b(4096, 4), decayBeta = b(32, 4), core = b(2048);
        const gate = b(3584), up = b(3584), ff = b(3584), down = b(1024);
        const query = b(2048), key = b(512), value = b(512), queryGate = b(4096);
        const capacity = Math.ceil((length + maxTokens + 8) / 256) * 256;
        const scores = b(8 * capacity, 4), probabilities = b(8 * capacity);
        const partialValues = b(8 * 8 * 256, 4);
        const encoder = this.device.createCommandEncoder();
        const gemv = (src, name, dst, k, n) => {
            if (this.device.features.has('subgroups') && this.options.gemv !== 'shared' && n >= 512) {
                const code = 'enable subgroups;\n' + bind(0, 'x') + bind(1, 'w') + bind(2, 'y', 'f16', true) + `
                    var<workgroup> parts: array<f32,128>;
                    @compute @workgroup_size(128) fn main(@builtin(local_invocation_index) lane:u32, @builtin(workgroup_id) g:vec3<u32>,
                        @builtin(subgroup_size) size:u32, @builtin(subgroup_invocation_id) slane:u32) {
                        let row=g.x*4u;var a=0.0;var b=0.0;var c=0.0;var d=0.0;
                        for(var j=lane;j<${k}u;j+=128u){let value=f32(x[j]);
                            a+=value*f32(w[row*${k}u+j]);b+=value*f32(w[(row+1u)*${k}u+j]);
                            c+=value*f32(w[(row+2u)*${k}u+j]);d+=value*f32(w[(row+3u)*${k}u+j]);}
                        let sa=subgroupAdd(a);let sb=subgroupAdd(b);let sc=subgroupAdd(c);let sd=subgroupAdd(d);
                        if(slane==0u){let p=(lane/size)*4u;parts[p]=sa;parts[p+1u]=sb;parts[p+2u]=sc;parts[p+3u]=sd;}
                        workgroupBarrier();
                        if(lane<4u&&row+lane<${n}u){var value=0.0;for(var i=0u;i<128u/size;i++){value+=parts[i*4u+lane];}y[row+lane]=f16(value);}
                    }`;
                add('gemv ' + name, code, [src, weights(name), dst], Math.ceil(n / 4));
                return;
            }
            const code = bind(0, 'x') + bind(1, 'w') + bind(2, 'y', 'f16', true) + reduce + `
                @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) g: vec3<u32>) {
                    let row = g.x; var acc = 0.0;
                    for (var j = lane; j < ${k}u; j += 64u) { acc += f32(x[j]) * f32(w[row * ${k}u + j]); }
                    let result = sum(acc, lane, 64u); if (lane == 0u) { y[row] = f16(result); }
                }`;
            add('gemv ' + name, code, [src, weights(name), dst], n);
        };
        const rms = (src, name, dst, residual) => {
            const code = bind(0, 'x', 'f32', true) + bind(1, 'weight', 'f32') + bind(2, 'y', 'f16', true) +
                (residual ? bind(3, 'addend') : '') + reduce + `
                @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane: u32) {
                    var acc = 0.0;
                    for (var i = lane; i < 1024u; i += 256u) { ${residual ? 'x[i] += f32(addend[i]);' : ''} acc += x[i] * x[i]; }
                    let denominator = sqrt(sum(acc, lane, 256u) / 1024.0 + 1e-6);
                    for (var i = lane; i < 1024u; i += 256u) { y[i] = f16((x[i] / denominator) * (1.0 + weight[i])); }
                }`;
            add('residual RMS', code, [src, weights(name), dst, ...(residual ? [residual] : [])], 1);
        };
        add('embedding', bind(0, 'embedding') + bind(1, 'params', 'u32') + bind(2, 'x', 'f32', true) + `
            @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) { x[id.x] = f32(embedding[params[0] * 1024u + id.x]); }
        `, [weights('embed_tokens.weight'), this.params, x], 4);
        for (let layer = 0; layer < 24; layer++) {
            const p = `layers.${layer}.`;
            // Fold the preceding MLP residual add into the next layer's normalization.
            rms(x, p + 'input_layernorm.weight', norm, layer ? down : null);
            if (layer % 4 !== 3) {
                const l = p + 'linear_attn.';
                const history = b(3 * 6144), state = b(16 * 128 * 128, 4);
                encoder.copyBufferToBuffer(cache[`conv_${layer}`].gpuBuffer, 0, history, 0, 3 * 6144 * 2);
                encoder.copyBufferToBuffer(cache[`state_${layer}`].gpuBuffer, 0, state, 0, 16 * 128 * 128 * 4);
                gemv(norm, l + 'in_proj_qkv.weight', qkv, 1024, 6144);
                gemv(norm, l + 'in_proj_z.weight', z, 1024, 2048);
                gemv(norm, l + 'in_proj_a.weight', a, 1024, 16);
                gemv(norm, l + 'in_proj_b.weight', beta, 1024, 16);
                add('causal conv SiLU', bind(0, 'x') + bind(1, 'history', 'f16', true) + bind(2, 'w', 'f32') + bind(3, 'y', 'f16', true) + `
                    @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                        let i = id.x; let v0 = history[i]; let v1 = history[6144u+i]; let v2 = history[12288u+i]; let v3 = x[i];
                        let v = (f32(v0)*w[i*4u] + f32(v1)*w[i*4u+1u]) + (f32(v2)*w[i*4u+2u] + f32(v3)*w[i*4u+3u]);
                        y[i] = f16(v * (1.0 / (1.0+exp(-v)))); history[i] = v1; history[6144u+i] = v2; history[12288u+i] = v3;
                    }`, [qkv, history, weights(l + 'conv1d.weight'), mixed], 24);
                add('qk norm decay', bind(0, 'x') + bind(1, 'a') + bind(2, 'b') + bind(3, 'dt', 'f32') + bind(4, 'alog', 'f32') +
                    bind(5, 'qk', 'f32', true) + bind(6, 'params', 'f32', true) + reduce + `
                    @compute @workgroup_size(128) fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) g: vec3<u32>) {
                        let h = g.x; let i = h*128u+lane; let q = f32(x[i]); let k = f32(x[2048u+i]);
                        let sq = sum(q*q, lane, 128u); let qnorm = (q / sqrt(sq+1e-6)) * 0.08838834764831845;
                        workgroupBarrier(); let sk = sum(k*k, lane, 128u);
                        qk[i] = qnorm; qk[2048u+i] = k / sqrt(sk+1e-6);
                        if (lane == 0u) { let v = f32(a[h])+dt[h]; let softplus = max(v,0.0)+log(1.0+exp(-abs(v)));
                            params[h] = exp(-exp(alog[h])*softplus); params[16u+h] = 1.0/(1.0+exp(-f32(b[h]))); }
                    }`, [mixed, a, beta, weights(l + 'dt_bias'), weights(l + 'A_log'), qk, decayBeta], 16);
                add('DeltaNet state RMS gate', bind(0, 'qk', 'f32') + bind(1, 'v') + bind(2, 'params', 'f32') + bind(3, 'state', 'f32', true) +
                    bind(4, 'z') + bind(5, 'w', 'f32') + bind(6, 'out', 'f16', true) + reduce + `
                    @compute @workgroup_size(128) fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) g: vec3<u32>) {
                        let h=g.x; let d=params[h]; var kv=0.0;
                        for (var j=0u; j<128u; j++) { kv += qk[2048u+h*128u+j] * (state[h*16384u+j*128u+lane]*d); }
                        let delta=(f32(v[4096u+h*128u+lane])-kv)*params[16u+h]; var output=0.0;
                        for (var j=0u; j<128u; j++) { let i=h*16384u+j*128u+lane;
                            let s=state[i]*d+qk[2048u+h*128u+j]*delta; state[i]=s; output += qk[h*128u+j]*s; }
                        let denominator=sqrt(sum(output*output,lane,128u)/128.0+1e-6); let gate=f32(z[h*128u+lane]);
                        out[h*128u+lane]=f16((output/denominator)*w[lane]*(gate*(1.0/(1.0+exp(-gate)))));
                    }`, [qk, mixed, decayBeta, state, z, weights(l + 'norm.weight'), core], 16);
                gemv(core, l + 'out_proj.weight', attended, 2048, 1024);
            } else {
                const l = p + 'self_attn.';
                const keys = b(2 * capacity * 256), values = b(2 * capacity * 256);
                for (let h = 0; h < 2; h++) for (const [keyName, dst] of [['key', keys], ['value', values]]) {
                    encoder.copyBufferToBuffer(cache[`${keyName}_${layer}`].gpuBuffer, h * length * 512, dst, h * capacity * 512, length * 512);
                }
                gemv(norm, l + 'q_proj.weight', queryGate, 1024, 4096);
                gemv(norm, l + 'k_proj.weight', key, 1024, 512);
                gemv(norm, l + 'v_proj.weight', value, 1024, 512);
                for (const isKey of [false, true]) {
                    const stride = isKey ? 256 : 512;
                    add(isKey ? 'key RMS RoPE cache' : 'query RMS RoPE', bind(0, 'x') + bind(1, 'w', 'f32') + bind(2, 'params', 'u32') +
                        bind(3, 'out', 'f16', true) + (isKey ? bind(4, 'value') + bind(5, 'values', 'f16', true) : '') + bind(isKey ? 6 : 4, 'rotary', 'f32') + reduce + `
                        var<workgroup> normalized: array<f32,256>;
                        @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) g: vec3<u32>) {
                            let h=g.x; let v=f32(x[h*${stride}u+lane]);
                            let denominator=sqrt(sum(v*v,lane,256u)/256.0+1e-6);
                            normalized[lane]=f32(f16((v/denominator)*(1.0+w[lane]))); workgroupBarrier(); var output=normalized[lane];
                            if (lane<64u) {
                                let rotated=select(-normalized[lane+32u],normalized[lane-32u],lane>=32u);
                                output=output*rotary[params[3]*128u+lane]+rotated*rotary[params[3]*128u+64u+lane]; }
                            let index=${isKey ? `h*${capacity * 256}u+params[2]*256u+lane` : 'h*256u+lane'};
                            out[index]=f16(output); ${isKey ? 'values[index]=value[h*256u+lane];' : ''}
                        }`, [isKey ? key : queryGate, weights(l + (isKey ? 'k_norm.weight' : 'q_norm.weight')), this.params,
                        isKey ? keys : query, ...(isKey ? [value, values] : []), rotary], isKey ? 2 : 8);
                }
                add('attention scores', bind(0, 'q') + bind(1, 'keys') + bind(2, 'params', 'u32') + bind(3, 'scores', 'f32', true) + reduce + `
                    @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>) {
                        let pos=g.x; let h=g.y; if(pos>params[2]) { return; } var acc=0.0;
                        for(var j=lane;j<256u;j+=64u){acc+=f32(q[h*256u+j])*f32(keys[(h/4u)*${capacity * 256}u+pos*256u+j]);}
                        let score=f16(sum(acc,lane,64u)); if(lane==0u){scores[h*${capacity}u+pos]=f32(score*f16(0.0625));}
                    }`, [query, keys, this.params, scores], [capacity, 8]);
                add('attention softmax', bind(0, 'scores', 'f32') + bind(1, 'params', 'u32') + bind(2, 'prob', 'f16', true) + reduce + `
                    @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>) {
                        let base=g.x*${capacity}u; var maximum=-3.4e38;
                        for(var j=lane;j<=params[2];j+=256u){maximum=max(maximum,scores[base+j]);}
                        sums[lane]=maximum;workgroupBarrier();
                        for(var s=128u;s>0u;s/=2u){if(lane<s){sums[lane]=max(sums[lane],sums[lane+s]);}workgroupBarrier();}
                        maximum=sums[0];workgroupBarrier();var acc=0.0;
                        for(var j=lane;j<=params[2];j+=256u){acc+=exp(scores[base+j]-maximum);}
                        let denominator=sum(acc,lane,256u);
                        for(var j=lane;j<=params[2];j+=256u){prob[base+j]=f16(exp(scores[base+j]-maximum)/denominator);}
                    }`, [scores, this.params, probabilities], 8);
                add('attention values split', bind(0, 'prob') + bind(1, 'values') + bind(2, 'params', 'u32') + bind(3, 'out', 'f32', true) + `
                    @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>) {
                        let h=g.x;var acc=0.0;
                        let count=(params[2]+8u)/8u;let end=min((g.y+1u)*count,params[2]+1u);
                        for(var j=g.y*count;j<end;j++){acc+=f32(prob[h*${capacity}u+j])*f32(values[(h/4u)*${capacity * 256}u+j*256u+lane]);}
                        out[(h*8u+g.y)*256u+lane]=acc;
                    }`, [probabilities, values, this.params, partialValues], [8, 8]);
                add('attention values gate', bind(0, 'parts', 'f32') + bind(1, 'qgate') + bind(2, 'out', 'f16', true) + `
                    @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>) {
                        let h=g.x;var acc=0.0;for(var i=0u;i<8u;i++){acc+=parts[(h*8u+i)*256u+lane];}
                        // ONNX Sigmoid has an FP16 input/output here. Preserve its intermediate rounding.
                        let gate=f16(1.0)/(f16(1.0)+exp(-qgate[h*512u+256u+lane]));out[h*256u+lane]=f16(acc)*gate;
                    }`, [partialValues, queryGate, core], 8);
                gemv(core, l + 'o_proj.weight', attended, 2048, 1024);
            }
            rms(x, p + 'post_attention_layernorm.weight', norm, attended);
            gemv(norm, p + 'mlp.gate_proj.weight', gate, 1024, 3584);
            gemv(norm, p + 'mlp.up_proj.weight', up, 1024, 3584);
            add('SwiGLU', bind(0, 'gate') + bind(1, 'up') + bind(2, 'out', 'f16', true) + `
                @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;let v=f32(gate[i]);out[i]=f16(v*(1.0/(1.0+exp(-v)))*f32(up[i]));}
            `, [gate, up, ff], 14);
            gemv(ff, p + 'mlp.down_proj.weight', down, 3584, 1024);
        }
        rms(x, 'norm.weight', norm, down);
        const logits = b(65425), maxima = b(256, 4), indices = b(256, 4);
        gemv(norm, 'embed_tokens.weight', logits, 1024, 65425);
        const argmax = (last = false) => bind(0, 'x', last ? 'f32' : 'f16') + bind(1, 'values', last ? 'u32' : 'f32', true) +
            bind(2, 'indices', 'u32', true) + (last ? bind(3, 'ids', 'u32') : '') + `
            var<workgroup> m:array<f32,256>;var<workgroup> ids:array<u32,256>;
            @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
                let i=g.x*256u+lane;var v=-3.4e38;var id=4294967295u;
                if(i<${last ? 256 : 65425}u){v=f32(x[i]);id=${last ? 'inputIds[i]' : 'i'};}
                m[lane]=v;ids[lane]=id;workgroupBarrier();
                for(var s=128u;s>0u;s/=2u){if(lane<s){if(m[lane+s]>m[lane]||(m[lane+s]==m[lane]&&ids[lane+s]<ids[lane])){m[lane]=m[lane+s];ids[lane]=ids[lane+s];}}workgroupBarrier();}
                if(lane==0u){${last ? 'indices[values[3]]=ids[0]; values[0]=ids[0];values[1]++;values[2]++;values[3]++;' : 'values[g.x]=m[0];indices[g.x]=ids[0];'}}
            }`;
        add('argmax blocks', argmax(), [logits, maxima, indices], 256);
        add('argmax token', argmax(true).replace('var<storage, read> ids:', 'var<storage, read> inputIds:'), [maxima, this.params, this.tokens, indices], 1);
        this.tasks = tasks; this.sequenceBuffers = this.buffers.slice(before);
        this.generatedCount = 0;
        this.readbacks = 0;
        this.profile = {};
        if (this.options.profile && this.device.features.has('timestamp-query')) {
            this.querySet = this.device.createQuerySet({ type: 'timestamp', count: tasks.length * 2 });
            this.queryBuffer = this.device.createBuffer({ size: tasks.length * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
            this.queryReadback = this.device.createBuffer({ size: tasks.length * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        }
        this.device.queue.submit([encoder.finish()]);
        const error = await this.device.popErrorScope();
        if (error) throw new Error(`Décodeur WebGPU : ${error.message}`);
    }
    async step(count = 1) {
        if (this.deviceLost) throw new Error(this.deviceLost);
        if (!Number.isInteger(count) || count < 1 || count > 8 || (this.querySet && count !== 1)) throw new Error('Taille de soumission GPU invalide.');
        const encoder = this.device.createCommandEncoder();
        if (this.querySet) {
            this.tasks.forEach((task, i) => {
                const pass = encoder.beginComputePass({ timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 } });
                this.dispatch(pass, task); pass.end();
            });
            encoder.resolveQuerySet(this.querySet, 0, this.tasks.length * 2, this.queryBuffer, 0);
            encoder.copyBufferToBuffer(this.queryBuffer, 0, this.queryReadback, 0, this.tasks.length * 16);
        } else {
            const pass = encoder.beginComputePass();
            for (let i = 0; i < count; i++) for (const task of this.tasks) this.dispatch(pass, task);
            pass.end();
        }
        encoder.copyBufferToBuffer(this.params, 0, this.readback, 0, 16);
        encoder.copyBufferToBuffer(this.tokens, this.generatedCount * 4, this.readback, 16, count * 4);
        this.device.queue.submit([encoder.finish()]);
        await this.readback.mapAsync(GPUMapMode.READ, 0, 16 + count * 4);
        const next = Array.from(new Uint32Array(this.readback.getMappedRange(0, 16 + count * 4)).slice(4));
        this.readback.unmap();
        this.generatedCount += count;
        this.readbacks++;
        if (this.querySet) {
            await this.queryReadback.mapAsync(GPUMapMode.READ);
            const times = new BigUint64Array(this.queryReadback.getMappedRange());
            this.tasks.forEach((task, i) => {
                const name = task.label.startsWith('gemv ') ? task.label.split('.').slice(-2).join('.') : task.label;
                const entry = this.profile[name] ||= { calls: 0, gpuMs: 0 };
                entry.calls++; entry.gpuMs += Number(times[i * 2 + 1] - times[i * 2]) / 1e6;
            });
            this.queryReadback.unmap();
        }
        return next;
    }
    end() {
        const discarded = new Set(this.sequenceStart === undefined ? [] : this.buffers.slice(this.sequenceStart));
        discarded.forEach(b => b.destroy()); this.buffers = this.buffers.filter(b => !discarded.has(b));
        this.sequenceStart = undefined;
        this.readback?.destroy(); this.readback = null; this.sequenceBuffers = []; this.tasks = [];
        this.querySet?.destroy(); this.queryBuffer?.destroy(); this.queryReadback?.destroy(); this.querySet = null;
    }
    stats() {
        return { kernelsPerStep: this.tasks.length, computedSteps: this.generatedCount, readbacks: this.readbacks,
            weightBytes: this.weightBytes, sequenceBytes: (this.sequenceBuffers || []).reduce((bytes, b) => bytes + b.size, 0),
            subgroups: this.device.features.has('subgroups') && this.options.gemv !== 'shared' };
    }
    dispose() { this.end(); this.buffers.forEach(b => b.destroy()); this.buffers = []; this.weights = {}; }
}
