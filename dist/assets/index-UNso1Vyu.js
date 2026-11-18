var ge=Object.defineProperty;var me=(s,e,t)=>e in s?ge(s,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):s[e]=t;var g=(s,e,t)=>me(s,typeof e!="symbol"?e+"":e,t);(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const o of document.querySelectorAll('link[rel="modulepreload"]'))r(o);new MutationObserver(o=>{for(const i of o)if(i.type==="childList")for(const n of i.addedNodes)n.tagName==="LINK"&&n.rel==="modulepreload"&&r(n)}).observe(document,{childList:!0,subtree:!0});function t(o){const i={};return o.integrity&&(i.integrity=o.integrity),o.referrerPolicy&&(i.referrerPolicy=o.referrerPolicy),o.crossOrigin==="use-credentials"?i.credentials="include":o.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function r(o){if(o.ep)return;o.ep=!0;const i=t(o);fetch(o.href,i)}})();async function ve(s){if(!navigator.gpu)throw new Error("当前浏览器不支持 WebGPU。请使用 Chrome / Edge 113+ 并确认已启用硬件加速。");const e=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!e)throw new Error("找不到可用的 GPU 适配器。");const t=i=>(i/1048576).toFixed(0)+" MB";if(e.limits.maxStorageBufferBindingSize<s)throw new Error(`显卡的 maxStorageBufferBindingSize 为 ${t(e.limits.maxStorageBufferBindingSize)},小于求解器所需的 ${t(s)}。请更换显卡或减小网格规模。`);const r={maxStorageBufferBindingSize:s};e.limits.maxBufferSize>=s&&(r.maxBufferSize=s);const o=await e.requestDevice({requiredLimits:r});return o.lost.then(i=>{i.reason!=="destroyed"&&console.error("WebGPU device lost:",i.message)}),o}const $=[[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,0],[-1,-1,0],[1,-1,0],[-1,1,0],[1,0,1],[-1,0,-1],[1,0,-1],[-1,0,1],[0,1,1],[0,-1,-1],[0,1,-1],[0,-1,1]],ye=[1/3,...Array(6).fill(1/18),...Array(12).fill(1/36)],Z=(s,e,t)=>$.findIndex(r=>r[0]===s&&r[1]===e&&r[2]===t),Be=$.map(s=>Z(-s[0],-s[1],-s[2])),xe=$.map(s=>Z(s[0],-s[1],s[2])),be=$.map(s=>Z(s[0],s[1],-s[2])),se=0,ne=1,Pe=2,we=3,k=4,O=4096;function Ce(s,e,t){const r=o=>o.map(i=>i+"u").join(",");return`
const NX: u32 = ${s}u;
const NY: u32 = ${e}u;
const NZ: u32 = ${t}u;
const NCELLS: u32 = ${s*e*t}u;
const C_DIRS = array<vec3i, 19>(${$.map(o=>`vec3i(${o[0]},${o[1]},${o[2]})`).join(",")});
const W = array<f32, 19>(${ye.map(o=>o.toFixed(8)).join(",")});
const OPP = array<u32, 19>(${r(Be)});
const MIRY = array<u32, 19>(${r(xe)});
const MIRZ = array<u32, 19>(${r(be)});
fn cellId(x: u32, y: u32, z: u32) -> u32 { return x + y * NX + z * NX * NY; }
fn feq(i: u32, rho: f32, u: vec3f) -> f32 {
  let c = vec3f(C_DIRS[i]);
  let cu = dot(c, u);
  return W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * dot(u, u));
}
`}class Se{constructor(e,t,r,o){g(this,"nx");g(this,"ny");g(this,"nz");g(this,"cells");g(this,"device");g(this,"fA");g(this,"fB");g(this,"flagsBuf");g(this,"macroBuf");g(this,"forceBuf");g(this,"staging");g(this,"stagingFree",[!0,!0]);g(this,"paramsBuf");g(this,"initListBuf");g(this,"initCountBuf");g(this,"INIT_CAP",262144);g(this,"stepPipe");g(this,"boundaryPipe");g(this,"macroPipe");g(this,"initPipe");g(this,"initCellsPipe");g(this,"stepBG");g(this,"boundaryBG");g(this,"macroBG");g(this,"cur",0);g(this,"flagsCPU");g(this,"frameParity",0);g(this,"u0",.08);g(this,"nu",8e-4);g(this,"CSMAG",.15);g(this,"lastSteps",0);this.device=e,this.nx=t,this.ny=r,this.nz=o,this.cells=t*r*o;const i=this.cells*19*4,n=(p,h)=>e.createBuffer({size:p,usage:h});this.fA=n(i,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),this.fB=n(i,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),this.flagsBuf=n(this.cells*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),this.macroBuf=n(this.cells*16,GPUBufferUsage.STORAGE),this.forceBuf=n(16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC),this.staging=[0,1].map(()=>n(16,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST)),this.paramsBuf=n(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),this.initListBuf=n(this.INIT_CAP*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),this.initCountBuf=n(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);const f=Ce(t,r,o),c=f+`
struct SimParams { tau: f32, u0: f32, css: f32, cells: u32 };
@group(0) @binding(0) var<storage, read> fIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> fOut: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> force: array<atomic<i32>>;
@group(0) @binding(4) var<uniform> params: SimParams;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  if (id >= NCELLS) { return; }
  if (flags[id] != 0u) { return; } // 只算流体格子
  let base = id * 19u;
  var f: array<f32, 19>;
  var rho = 0.0;
  var mom = vec3f(0.0);
  for (var i = 0u; i < 19u; i++) {
    f[i] = fIn[base + i];
    rho += f[i];
    mom += vec3f(C_DIRS[i]) * f[i];
  }
  let u = mom / rho;
  var fe: array<f32, 19>;
  for (var i = 0u; i < 19u; i++) { fe[i] = feq(i, rho, u); }
  // 非平衡应力二阶矩 → Smagorinsky 有效松弛时间
  var pxx = 0.0; var pyy = 0.0; var pzz = 0.0;
  var pxy = 0.0; var pxz = 0.0; var pyz = 0.0;
  for (var i = 0u; i < 19u; i++) {
    let fn = f[i] - fe[i];
    let c = vec3f(C_DIRS[i]);
    pxx += fn * c.x * c.x; pyy += fn * c.y * c.y; pzz += fn * c.z * c.z;
    pxy += fn * c.x * c.y; pxz += fn * c.x * c.z; pyz += fn * c.y * c.z;
  }
  let piMag = sqrt(2.0 * (pxx * pxx + pyy * pyy + pzz * pzz) + 4.0 * (pxy * pxy + pxz * pxz + pyz * pyz));
  let tau0 = params.tau;
  let tauEff = 0.5 * (tau0 + sqrt(tau0 * tau0 + params.css * piMag / rho));
  let omega = 1.0 / tauEff;

  let x = i32(id % NX);
  let y = i32((id / NX) % NY);
  let z = i32(id / (NX * NY));
  var forceV = vec3f(0.0);
  for (var i = 0u; i < 19u; i++) {
    let fp = f[i] - omega * (f[i] - fe[i]);
    let ci = C_DIRS[i];
    let nx = x + ci.x; let ny = y + ci.y; let nz = z + ci.z;
    if (nx < 0 || nx >= i32(NX) || ny < 0 || ny >= i32(NY) || nz < 0 || nz >= i32(NZ)) { continue; }
    let nid = cellId(u32(nx), u32(ny), u32(nz));
    if (flags[nid] == 1u) {
      // 固体:反弹回自己,动量差即作用力
      fOut[base + OPP[i]] = fp;
      forceV += 2.0 * vec3f(ci) * fp;
    } else {
      fOut[nid * 19u + i] = fp;
    }
  }
  if (forceV.x != 0.0 || forceV.y != 0.0 || forceV.z != 0.0) {
    atomicAdd(&force[0], i32(forceV.x * ${O}.0));
    atomicAdd(&force[1], i32(forceV.y * ${O}.0));
    atomicAdd(&force[2], i32(forceV.z * ${O}.0));
  }
}`,u=f+`
struct SimParams { tau: f32, u0: f32, css: f32, cells: u32 };
@group(0) @binding(0) var<storage, read> fIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> fOut: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  if (id >= NCELLS) { return; }
  let x = id % NX;
  let y = (id / NX) % NY;
  let z = id / (NX * NY);
  let base = id * 19u;
  let uIn = vec3f(params.u0, 0.0, 0.0);
  if (x == 0u) { // 入口:平衡态
    for (var i = 0u; i < 19u; i++) { fOut[base + i] = feq(i, 1.0, uIn); }
    return;
  }
  if (x == NX - 1u) { // 出口:零梯度(拷贝上游邻居)
    let src = base - 19u;
    for (var i = 0u; i < 19u; i++) { fOut[base + i] = fIn[src + i]; }
    return;
  }
  if (flags[id] != 0u) { return; }
  // 入口旁的流体格子:指向下游的分布没人写入,用入口平衡态补
  if (x == 1u) {
    for (var i = 0u; i < 19u; i++) {
      if (C_DIRS[i].x > 0) { fOut[base + i] = feq(i, 1.0, uIn); }
    }
  }
  // 出口旁的流体格子:指向上游的分布用自身宏观量的平衡态补
  if (x == NX - 2u) {
    var rho = 0.0; var mom = vec3f(0.0);
    for (var i = 0u; i < 19u; i++) {
      rho += fIn[base + i];
      mom += vec3f(C_DIRS[i]) * fIn[base + i];
    }
    let u = mom / rho;
    for (var i = 0u; i < 19u; i++) {
      if (C_DIRS[i].x < 0) { fOut[base + i] = feq(i, rho, u); }
    }
  }
  // 滑移壁面:法向分量镜面反射(读主 kernel 写好的同格分布)
  if (y == 1u) {
    for (var i = 0u; i < 19u; i++) {
      if (C_DIRS[i].y > 0) { fOut[base + i] = fOut[base + MIRY[i]]; }
    }
  }
  if (y == NY - 2u) {
    for (var i = 0u; i < 19u; i++) {
      if (C_DIRS[i].y < 0) { fOut[base + i] = fOut[base + MIRY[i]]; }
    }
  }
  if (z == 1u) {
    for (var i = 0u; i < 19u; i++) {
      if (C_DIRS[i].z > 0) { fOut[base + i] = fOut[base + MIRZ[i]]; }
    }
  }
  if (z == NZ - 2u) {
    for (var i = 0u; i < 19u; i++) {
      if (C_DIRS[i].z < 0) { fOut[base + i] = fOut[base + MIRZ[i]]; }
    }
  }
}`,l=f+`
@group(0) @binding(0) var<storage, read> fIn: array<f32>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> macro: array<vec4f>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  if (id >= NCELLS) { return; }
  let fl = flags[id];
  if (fl == 1u || fl == 4u) {
    macro[id] = vec4f(0.0, 0.0, 0.0, 1.0);
    return;
  }
  let base = id * 19u;
  var rho = 0.0; var mom = vec3f(0.0);
  for (var i = 0u; i < 19u; i++) {
    let fi = fIn[base + i];
    rho += fi;
    mom += vec3f(C_DIRS[i]) * fi;
  }
  macro[id] = vec4f(mom / rho, rho);
}`,d=f+`
@group(0) @binding(0) var<storage, read_write> fA: array<f32>;
@group(0) @binding(1) var<storage, read_write> fB: array<f32>;
@group(0) @binding(2) var<uniform> u0u: vec4f;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  if (id >= NCELLS) { return; }
  let u = vec3f(u0u.x, 0.0, 0.0);
  for (var i = 0u; i < 19u; i++) {
    let v = feq(i, 1.0, u);
    fA[id * 19u + i] = v;
    fB[id * 19u + i] = v;
  }
}`,a=f+`
@group(0) @binding(0) var<storage, read_write> fA: array<f32>;
@group(0) @binding(1) var<storage, read_write> fB: array<f32>;
@group(0) @binding(2) var<storage, read> list: array<u32>;
@group(0) @binding(3) var<uniform> u0u: vec4f; // x: u0, y: count

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= u32(u0u.y)) { return; }
  let id = list[k];
  let u = vec3f(u0u.x, 0.0, 0.0);
  for (var i = 0u; i < 19u; i++) {
    let v = feq(i, 1.0, u);
    fA[id * 19u + i] = v;
    fB[id * 19u + i] = v;
  }
}`,m=p=>e.createComputePipeline({layout:"auto",compute:{module:e.createShaderModule({code:p}),entryPoint:"main"}});this.stepPipe=m(c),this.boundaryPipe=m(u),this.macroPipe=m(l),this.initPipe=m(d),this.initCellsPipe=m(a);const v=[this.fA,this.fB];this.stepBG=[0,1].map(p=>e.createBindGroup({layout:this.stepPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:v[p]}},{binding:1,resource:{buffer:v[1-p]}},{binding:2,resource:{buffer:this.flagsBuf}},{binding:3,resource:{buffer:this.forceBuf}},{binding:4,resource:{buffer:this.paramsBuf}}]})),this.boundaryBG=[0,1].map(p=>e.createBindGroup({layout:this.boundaryPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:v[p]}},{binding:1,resource:{buffer:v[1-p]}},{binding:2,resource:{buffer:this.flagsBuf}},{binding:3,resource:{buffer:this.paramsBuf}}]})),this.macroBG=[0,1].map(p=>e.createBindGroup({layout:this.macroPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:v[p]}},{binding:1,resource:{buffer:this.flagsBuf}},{binding:2,resource:{buffer:this.macroBuf}}]})),this.flagsCPU=new Uint32Array(this.cells),this.rebuildBoundaryFlags(),e.queue.writeBuffer(this.flagsBuf,0,this.flagsCPU),this.uploadParams(),this.resetFlow()}rebuildBoundaryFlags(){const{nx:e,ny:t,nz:r,flagsCPU:o}=this;o.fill(se);for(let i=0;i<r;i++)for(let n=0;n<t;n++)o[n*e+i*e*t]=Pe,o[e-1+n*e+i*e*t]=we;for(let i=0;i<r;i++)for(let n=0;n<e;n++)o[n+i*e*t]=k,o[n+(t-1)*e+i*e*t]=k;for(let i=0;i<t;i++)for(let n=0;n<e;n++)o[n+i*e]=k,o[n+i*e+(r-1)*e*t]=k}uploadParams(){const e=.5+3*this.nu,t=18*Math.SQRT2*this.CSMAG*this.CSMAG;this.device.queue.writeBuffer(this.paramsBuf,0,new Float32Array([e,this.u0,t,this.cells]))}setFlow(e,t){this.u0=e,this.nu=t,this.uploadParams()}resetFlow(){const e=this.device.createCommandEncoder(),t=e.beginComputePass();t.setPipeline(this.initPipe),t.setBindGroup(0,this.device.createBindGroup({layout:this.initPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.fA}},{binding:1,resource:{buffer:this.fB}},{binding:2,resource:{buffer:this.paramsBuf}}]})),t.dispatchWorkgroups(Math.ceil(this.cells/128)),t.end(),this.device.queue.submit([e.finish()]),this.cur=0}updateObstacles(e){const{nx:t,ny:r,nz:o}=this,i=[];for(let c=1;c<o-1;c++)for(let u=1;u<r-1;u++)for(let l=1;l<t-1;l++){const d=l+u*t+c*t*r,a=this.flagsCPU[d]===ne,m=e[d]===1;a&&!m&&i.push(d),this.flagsCPU[d]=m?ne:se}if(this.device.queue.writeBuffer(this.flagsBuf,0,this.flagsCPU),i.length===0)return;if(i.length>this.INIT_CAP){this.resetFlow();return}this.device.queue.writeBuffer(this.initListBuf,0,new Uint32Array(i)),this.device.queue.writeBuffer(this.initCountBuf,0,new Float32Array([this.u0,i.length,0,0]));const n=this.device.createCommandEncoder(),f=n.beginComputePass();f.setPipeline(this.initCellsPipe),f.setBindGroup(0,this.device.createBindGroup({layout:this.initCellsPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.fA}},{binding:1,resource:{buffer:this.fB}},{binding:2,resource:{buffer:this.initListBuf}},{binding:3,resource:{buffer:this.initCountBuf}}]})),f.dispatchWorkgroups(Math.ceil(i.length/64)),f.end(),this.device.queue.submit([n.finish()])}encodeSteps(e,t){this.device.queue.writeBuffer(this.forceBuf,0,new Int32Array(4));const r=Math.ceil(this.cells/128),o=e.beginComputePass();for(let i=0;i<t;i++)o.setPipeline(this.stepPipe),o.setBindGroup(0,this.stepBG[this.cur]),o.dispatchWorkgroups(r),o.setPipeline(this.boundaryPipe),o.setBindGroup(0,this.boundaryBG[this.cur]),o.dispatchWorkgroups(r),this.cur=1-this.cur;o.setPipeline(this.macroPipe),o.setBindGroup(0,this.macroBG[this.cur]),o.dispatchWorkgroups(r),o.end(),this.frameParity=1-this.frameParity,e.copyBufferToBuffer(this.forceBuf,0,this.staging[this.frameParity],0,16),this.lastSteps=t}async readForce(){const e=this.frameParity;if(!this.stagingFree[e])return null;this.stagingFree[e]=!1;const t=this.staging[e];await t.mapAsync(GPUMapMode.READ);const r=new Int32Array(t.getMappedRange().slice(0));t.unmap(),this.stagingFree[e]=!0;const o=Math.max(1,this.lastSteps);return{fx:r[0]/O/o,fy:r[1]/O/o,fz:r[2]/O/o}}}class Ge{constructor(e,t,r,o,i,n,f,c,u=2048,l=48){g(this,"NPART");g(this,"HIST");g(this,"device");g(this,"partBuf");g(this,"histBuf");g(this,"paramsBuf");g(this,"slotBuf");g(this,"advectPipe");g(this,"advectBG");g(this,"renderPipe");g(this,"renderBG");g(this,"writeSlot",0);g(this,"frame",0);this.nx=t,this.ny=r,this.nz=o,this.device=e,this.NPART=u,this.HIST=l,this.partBuf=e.createBuffer({size:u*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.histBuf=e.createBuffer({size:u*l*16,usage:GPUBufferUsage.STORAGE}),this.paramsBuf=e.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.slotBuf=e.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const d=new ArrayBuffer(u*32),a=new Float32Array(d),m=new Uint32Array(d);for(let h=0;h<u;h++){const B=h%128,y=B/16|0,P=B%16;a[h*8+0]=8+Math.random()*(t-30),a[h*8+1]=4+y*7+Math.random()*3,a[h*8+2]=20+P*3.6+Math.random()*2,a[h*8+3]=Math.random()*260,m[h*8+4]=B,m[h*8+5]=h*2654435761>>>0}e.queue.writeBuffer(this.partBuf,0,d);const v=`
const NX: u32 = ${t}u;
const NY: u32 = ${r}u;
const NZ: u32 = ${o}u;
const NPART: u32 = ${u}u;
const HIST: u32 = ${l}u;
struct Particle { pos: vec3f, age: f32, nozzle: u32, seed: u32, pad: f32 };
struct PParams { dt: f32, writeSlot: u32, maxAge: f32, frame: u32 };
@group(0) @binding(0) var<storage, read> macro: array<vec4f>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(3) var<storage, read_write> hist: array<vec4f>;
@group(0) @binding(4) var<uniform> pp: PParams;

fn cid(x: u32, y: u32, z: u32) -> u32 { return x + y * NX + z * NX * NY; }
fn hashU(n0: u32) -> u32 {
  var n = n0;
  n ^= n >> 16u; n *= 2246822519u; n ^= n >> 13u; n *= 3266489917u; n ^= n >> 16u;
  return n;
}
fn rand(s: u32) -> f32 { return f32(hashU(s) & 0xFFFFFFu) / 16777216.0; }

fn sampleU(p: vec3f) -> vec3f {
  let pc = clamp(p, vec3f(1.0), vec3f(f32(NX) - 2.001, f32(NY) - 2.001, f32(NZ) - 2.001));
  let i0 = vec3u(floor(pc));
  let fr = pc - floor(pc);
  var acc = vec3f(0.0);
  for (var dz = 0u; dz < 2u; dz++) {
    for (var dy = 0u; dy < 2u; dy++) {
      for (var dx = 0u; dx < 2u; dx++) {
        let w = select(1.0 - fr.x, fr.x, dx == 1u) * select(1.0 - fr.y, fr.y, dy == 1u) * select(1.0 - fr.z, fr.z, dz == 1u);
        acc += w * macro[cid(i0.x + dx, i0.y + dy, i0.z + dz)].xyz;
      }
    }
  }
  return acc;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= NPART) { return; }
  var p = parts[i];
  var pos = p.pos;
  // 子步进,防止大步长穿过车身
  var dt = min(pp.dt, 24.0);
  let nsub = max(1u, u32(ceil(dt / 4.0)));
  dt /= f32(nsub);
  for (var s = 0u; s < nsub; s++) {
    let u1 = sampleU(pos);
    let u2 = sampleU(pos + 0.5 * dt * u1);
    pos += dt * u2;
  }
  p.age += 1.0;
  var dead = pos.x > f32(NX - 4u) || pos.x < 1.0 || pos.y < 1.0 || pos.y > f32(NY - 2u) ||
             pos.z < 1.0 || pos.z > f32(NZ - 2u) || p.age > pp.maxAge;
  if (!dead) {
    let c = cid(u32(pos.x), u32(pos.y), u32(pos.z));
    if (flags[c] == 1u) { dead = true; }
  }
  if (dead) {
    let row = p.nozzle / 16u;
    let col = p.nozzle % 16u;
    let r1 = rand(p.seed + pp.frame * 747u);
    let r2 = rand(p.seed + pp.frame * 1511u + 7u);
    let r3 = rand(p.seed + pp.frame * 2281u + 13u);
    pos = vec3f(8.0 + r1 * 2.0, 4.0 + f32(row) * 7.0 + r2 * 3.0, 20.0 + f32(col) * 3.6 + r3 * 2.0);
    p.age = r2 * 12.0; // 错开重生
    p.seed = hashU(p.seed + pp.frame);
  }
  p.pos = pos;
  parts[i] = p;
  hist[i * HIST + pp.writeSlot] = vec4f(pos, 1.0);
}`;this.advectPipe=e.createComputePipeline({layout:"auto",compute:{module:e.createShaderModule({code:v}),entryPoint:"main"}}),this.advectBG=e.createBindGroup({layout:this.advectPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:n}},{binding:2,resource:{buffer:this.partBuf}},{binding:3,resource:{buffer:this.histBuf}},{binding:4,resource:{buffer:this.paramsBuf}}]});const p=`
const NPART: u32 = ${u}u;
const HIST: u32 = ${l}u;
struct Cam { vp: mat4x4f, eye: vec3f };
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> hist: array<vec4f>;
@group(0) @binding(2) var<uniform> slotU: vec4u;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) alpha: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  let seg = vi / 2u;
  let e = vi % 2u;
  let p = seg / (HIST - 1u);
  let s = seg % (HIST - 1u);
  let slotA = (slotU.x + HIST - s) % HIST;       // 较新
  let slotB = (slotU.x + HIST - s - 1u) % HIST;  // 较旧
  let A = hist[p * HIST + slotA].xyz;
  let B = hist[p * HIST + slotB].xyz;
  var wp = select(A, B, e == 1u);
  var alpha = 0.5 * (1.0 - f32(s) / f32(HIST));
  if (distance(A, B) > 2.5) { alpha = 0.0; wp = vec3f(0.0); } // 重生跳变,隐藏
  var o: VOut;
  o.pos = cam.vp * vec4f(wp, 1.0);
  o.alpha = alpha;
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  return vec4f(0.72, 0.82, 0.95, in.alpha);
}`;this.renderPipe=e.createRenderPipeline({layout:"auto",vertex:{module:e.createShaderModule({code:p}),entryPoint:"vs"},fragment:{module:e.createShaderModule({code:p}),entryPoint:"fs",targets:[{format:c,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}}}]},primitive:{topology:"line-list"},depthStencil:{format:"depth24plus",depthWriteEnabled:!1,depthCompare:"less"}}),this.renderBG=e.createBindGroup({layout:this.renderPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:this.histBuf}},{binding:2,resource:{buffer:this.slotBuf}}]})}encodeAdvect(e,t){this.writeSlot=(this.writeSlot+1)%this.HIST,this.frame++;const r=new ArrayBuffer(16),o=new Float32Array(r),i=new Uint32Array(r);o[0]=t,i[1]=this.writeSlot,o[2]=260,i[3]=this.frame,this.device.queue.writeBuffer(this.paramsBuf,0,r),this.device.queue.writeBuffer(this.slotBuf,0,new Uint32Array([this.writeSlot,0,0,0]));const n=e.beginComputePass();n.setPipeline(this.advectPipe),n.setBindGroup(0,this.advectBG),n.dispatchWorkgroups(Math.ceil(this.NPART/64)),n.end()}prime(){for(let e=0;e<this.HIST;e++){const t=this.device.createCommandEncoder();this.encodeAdvect(t,1),this.device.queue.submit([t.finish()])}}encodeRender(e){e.setPipeline(this.renderPipe),e.setBindGroup(0,this.renderBG),e.draw((this.HIST-1)*2*this.NPART)}}const ae=`
fn colormap(t0: f32) -> vec3f {
  let t = clamp(t0, 0.0, 1.0);
  var c = mix(vec3f(0.10, 0.20, 0.90), vec3f(0.10, 0.80, 0.90), clamp(t * 4.0, 0.0, 1.0));
  c = mix(c, vec3f(0.20, 0.85, 0.30), clamp(t * 4.0 - 1.0, 0.0, 1.0));
  c = mix(c, vec3f(0.98, 0.85, 0.15), clamp(t * 4.0 - 2.0, 0.0, 1.0));
  c = mix(c, vec3f(0.90, 0.15, 0.10), clamp(t * 4.0 - 3.0, 0.0, 1.0));
  return c;
}
`;class Ae{constructor(e,t,r,o,i,n,f,c){g(this,"device");g(this,"camBuf");g(this,"carVertBuf",null);g(this,"carIdxBuf",null);g(this,"carColorBuf",null);g(this,"carIdxCount",0);g(this,"carVertCount",0);g(this,"carPipe");g(this,"carBG",null);g(this,"colorPipe");g(this,"colorBG",null);g(this,"colorParamsBuf");g(this,"groundPipe");g(this,"groundBG");g(this,"slicePipe");g(this,"sliceBG");g(this,"sliceParamsBuf");g(this,"macroBuf");g(this,"flagsBuf");this.nx=t,this.ny=r,this.nz=o,this.device=e,this.camBuf=i,this.colorParamsBuf=e.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.sliceParamsBuf=e.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const u=`
struct Cam { vp: mat4x4f, eye: vec3f };
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> colors: array<vec4f>;
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
};
@vertex
fn vs(@location(0) pos: vec3f, @location(1) nrm: vec3f, @builtin(vertex_index) vi: u32) -> VOut {
  var o: VOut;
  o.pos = cam.vp * vec4f(pos, 1.0);
  o.color = colors[vi].rgb;
  o.nrm = nrm;
  return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let l = normalize(vec3f(0.45, 0.8, 0.35));
  let n = normalize(in.nrm);
  let diff = 0.35 + 0.65 * abs(dot(n, l));
  return vec4f(in.color * diff, 1.0);
}`;this.carPipe=e.createRenderPipeline({layout:"auto",vertex:{module:e.createShaderModule({code:u}),entryPoint:"vs",buffers:[{arrayStride:24,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"}]}]},fragment:{module:e.createShaderModule({code:u}),entryPoint:"fs",targets:[{format:c}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}});const l=`
const NX: u32 = ${t}u;
const NY: u32 = ${r}u;
const NZ: u32 = ${o}u;
@group(0) @binding(0) var<storage, read> macro: array<vec4f>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> verts: array<f32>;
@group(0) @binding(3) var<storage, read_write> colors: array<vec4f>;
@group(0) @binding(4) var<uniform> params: vec4f; // x: u0
${ae}
fn cid(x: u32, y: u32, z: u32) -> u32 { return x + y * NX + z * NX * NY; }
fn rhoAt(p: vec3f) -> f32 {
  let pc = clamp(p, vec3f(1.0), vec3f(f32(NX) - 2.001, f32(NY) - 2.001, f32(NZ) - 2.001));
  let i0 = vec3u(floor(pc));
  let fr = pc - floor(pc);
  var acc = 0.0;
  for (var dz = 0u; dz < 2u; dz++) {
    for (var dy = 0u; dy < 2u; dy++) {
      for (var dx = 0u; dx < 2u; dx++) {
        let w = select(1.0 - fr.x, fr.x, dx == 1u) * select(1.0 - fr.y, fr.y, dy == 1u) * select(1.0 - fr.z, fr.z, dz == 1u);
        acc += w * macro[cid(i0.x + dx, i0.y + dy, i0.z + dz)].w;
      }
    }
  }
  return acc;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let vi = gid.x;
  if (vi >= arrayLength(&colors)) { return; }
  let p = vec3f(verts[vi * 6u], verts[vi * 6u + 1u], verts[vi * 6u + 2u]);
  let n = vec3f(verts[vi * 6u + 3u], verts[vi * 6u + 4u], verts[vi * 6u + 5u]);
  // 沿法向探到流体里取样,避免取到固体内部
  var sp = p + n * 1.3;
  var c = cid(u32(sp.x), u32(sp.y), u32(sp.z));
  if (flags[c] == 1u) { sp = p + n * 2.6; }
  let u0 = params.x;
  let cp = (rhoAt(sp) - 1.0) / (1.5 * u0 * u0);
  colors[vi] = vec4f(colormap(cp * 0.45 + 0.5), 1.0);
}`;this.colorPipe=e.createComputePipeline({layout:"auto",compute:{module:e.createShaderModule({code:l}),entryPoint:"main"}}),this.macroBuf=n,this.flagsBuf=f;const d=`
struct Cam { vp: mat4x4f, eye: vec3f };
@group(0) @binding(0) var<uniform> cam: Cam;
struct VOut { @builtin(position) pos: vec4f, @location(0) wp: vec3f };
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(${t}.0, 0.0), vec2f(${t}.0, ${o}.0),
    vec2f(0.0, 0.0), vec2f(${t}.0, ${o}.0), vec2f(0.0, ${o}.0));
  let c = corners[vi];
  var o: VOut;
  o.wp = vec3f(c.x, 0.5, c.y);
  o.pos = cam.vp * vec4f(o.wp, 1.0);
  return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let g = 8.0;
  let lx = abs(fract(in.wp.x / g) - 0.5);
  let lz = abs(fract(in.wp.z / g) - 0.5);
  let line = step(0.47, max(lx, lz));
  let base = vec3f(0.10, 0.115, 0.14);
  return vec4f(base + line * vec3f(0.05, 0.06, 0.07), 1.0);
}`;this.groundPipe=e.createRenderPipeline({layout:"auto",vertex:{module:e.createShaderModule({code:d}),entryPoint:"vs"},fragment:{module:e.createShaderModule({code:d}),entryPoint:"fs",targets:[{format:c}]},primitive:{topology:"triangle-list"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),this.groundBG=e.createBindGroup({layout:this.groundPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:i}}]});const a=`
const NX: u32 = ${t}u;
const NY: u32 = ${r}u;
const NZ: u32 = ${o}u;
struct Cam { vp: mat4x4f, eye: vec3f };
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> macro: array<vec4f>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<uniform> sp: vec4f; // x: 切片 z 坐标, y: u0
${ae}
fn cid(x: u32, y: u32, z: u32) -> u32 { return x + y * NX + z * NX * NY; }
struct VOut { @builtin(position) pos: vec4f, @location(0) wp: vec3f };
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(f32(NX), 0.0), vec2f(f32(NX), f32(NY)),
    vec2f(0.0, 0.0), vec2f(f32(NX), f32(NY)), vec2f(0.0, f32(NY)));
  let c = corners[vi];
  var o: VOut;
  o.wp = vec3f(c.x, c.y, sp.x);
  o.pos = cam.vp * vec4f(o.wp, 1.0);
  return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let x = u32(clamp(in.wp.x, 0.0, f32(NX) - 1.0));
  let y = u32(clamp(in.wp.y, 0.0, f32(NY) - 1.0));
  let z = u32(clamp(in.wp.z, 0.0, f32(NZ) - 1.0));
  let id = cid(x, y, z);
  if (flags[id] == 1u) { return vec4f(0.05, 0.05, 0.06, 0.95); }
  let u = macro[id].xyz;
  let speed = length(u) / sp.y;
  return vec4f(colormap(speed * 0.7), 0.82);
}`;this.slicePipe=e.createRenderPipeline({layout:"auto",vertex:{module:e.createShaderModule({code:a}),entryPoint:"vs"},fragment:{module:e.createShaderModule({code:a}),entryPoint:"fs",targets:[{format:c,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}}}]},primitive:{topology:"triangle-list"},depthStencil:{format:"depth24plus",depthWriteEnabled:!1,depthCompare:"less"}}),this.sliceBG=e.createBindGroup({layout:this.slicePipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:n}},{binding:2,resource:{buffer:f}},{binding:3,resource:{buffer:this.sliceParamsBuf}}]})}setMesh(e){var o,i,n;const t=e.positions.length/3,r=new Float32Array(t*6);for(let f=0;f<t;f++)r[f*6]=e.positions[f*3],r[f*6+1]=e.positions[f*3+1],r[f*6+2]=e.positions[f*3+2],r[f*6+3]=e.normals[f*3],r[f*6+4]=e.normals[f*3+1],r[f*6+5]=e.normals[f*3+2];(o=this.carVertBuf)==null||o.destroy(),(i=this.carIdxBuf)==null||i.destroy(),(n=this.carColorBuf)==null||n.destroy(),this.carVertBuf=this.device.createBuffer({size:r.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this.carVertBuf,0,r),this.carIdxBuf=this.device.createBuffer({size:e.indices.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this.carIdxBuf,0,e.indices),this.carColorBuf=this.device.createBuffer({size:t*16,usage:GPUBufferUsage.STORAGE}),this.carIdxCount=e.indices.length,this.carVertCount=t,this.carBG=this.device.createBindGroup({layout:this.carPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.camBuf}},{binding:1,resource:{buffer:this.carColorBuf}}]}),this.colorBG=this.device.createBindGroup({layout:this.colorPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.macroBuf}},{binding:1,resource:{buffer:this.flagsBuf}},{binding:2,resource:{buffer:this.carVertBuf}},{binding:3,resource:{buffer:this.carColorBuf}},{binding:4,resource:{buffer:this.colorParamsBuf}}]})}encodeCarColors(e,t){if(!this.colorBG)return;this.device.queue.writeBuffer(this.colorParamsBuf,0,new Float32Array([t,0,0,0]));const r=e.beginComputePass();r.setPipeline(this.colorPipe),r.setBindGroup(0,this.colorBG),r.dispatchWorkgroups(Math.ceil(this.carVertCount/64)),r.end()}setSlice(e,t){this.device.queue.writeBuffer(this.sliceParamsBuf,0,new Float32Array([e*(this.nz-1),t,0,0]))}encodeGround(e){e.setPipeline(this.groundPipe),e.setBindGroup(0,this.groundBG),e.draw(6)}encodeSlice(e){e.setPipeline(this.slicePipe),e.setBindGroup(0,this.sliceBG),e.draw(6)}encodeCar(e){!this.carBG||!this.carVertBuf||!this.carIdxBuf||(e.setPipeline(this.carPipe),e.setBindGroup(0,this.carBG),e.setVertexBuffer(0,this.carVertBuf),e.setIndexBuffer(this.carIdxBuf,"uint32"),e.drawIndexed(this.carIdxCount))}}function j(){const s=new Float32Array(16);return s[0]=s[5]=s[10]=s[15]=1,s}function de(s,e){const t=new Float32Array(16);for(let r=0;r<4;r++)for(let o=0;o<4;o++)t[r*4+o]=s[o]*e[r*4]+s[4+o]*e[r*4+1]+s[8+o]*e[r*4+2]+s[12+o]*e[r*4+3];return t}function Ue(s,e,t,r){const o=1/Math.tan(s/2),i=new Float32Array(16);return i[0]=o/e,i[5]=o,i[10]=r/(t-r),i[11]=-1,i[14]=r*t/(t-r),i}function Me(s,e){return[s[0]-e[0],s[1]-e[1],s[2]-e[2]]}function ue(s,e){return[s[1]*e[2]-s[2]*e[1],s[2]*e[0]-s[0]*e[2],s[0]*e[1]-s[1]*e[0]]}function W(s,e){return s[0]*e[0]+s[1]*e[1]+s[2]*e[2]}function ce(s){const e=Math.hypot(s[0],s[1],s[2])||1;return[s[0]/e,s[1]/e,s[2]/e]}function ze(s,e,t){const r=ce(Me(s,e)),o=ce(ue(t,r)),i=ue(r,o),n=j();return n[0]=o[0],n[4]=o[1],n[8]=o[2],n[1]=i[0],n[5]=i[1],n[9]=i[2],n[2]=r[0],n[6]=r[1],n[10]=r[2],n[12]=-W(o,s),n[13]=-W(i,s),n[14]=-W(r,s),n}function Fe(s,e,t,r){const o=s+s,i=e+e,n=t+t,f=s*o,c=s*i,u=s*n,l=e*i,d=e*n,a=t*n,m=r*o,v=r*i,p=r*n,h=j();return h[0]=1-(l+a),h[1]=c+p,h[2]=u-v,h[4]=c-p,h[5]=1-(f+a),h[6]=d+m,h[8]=u+v,h[9]=d-m,h[10]=1-(f+l),h}function Te(s,e,t){const r=Fe(e[0],e[1],e[2],e[3]);return r[0]*=t[0],r[1]*=t[0],r[2]*=t[0],r[4]*=t[1],r[5]*=t[1],r[6]*=t[1],r[8]*=t[2],r[9]*=t[2],r[10]*=t[2],r[12]=s[0],r[13]=s[1],r[14]=s[2],r}function Ie(s,e,t,r){return[s[0]*e+s[4]*t+s[8]*r+s[12],s[1]*e+s[5]*t+s[9]*r+s[13],s[2]*e+s[6]*t+s[10]*r+s[14]]}class Oe{constructor(e){g(this,"target",[100,18,48]);g(this,"dist",240);g(this,"theta",-.7);g(this,"phi",.32);g(this,"canvas");this.canvas=e;let t=0,r=0,o=0;e.addEventListener("pointerdown",n=>{t=n.button===0?1:2,r=n.clientX,o=n.clientY,e.setPointerCapture(n.pointerId)}),e.addEventListener("pointermove",n=>{if(!t)return;const f=n.clientX-r,c=n.clientY-o;if(r=n.clientX,o=n.clientY,t===1)this.theta-=f*.006,this.phi=Math.min(1.45,Math.max(-.2,this.phi+c*.006));else{const u=this.dist*.0012,l=[Math.cos(this.theta),0,-Math.sin(this.theta)];this.target[0]-=l[0]*f*u,this.target[2]-=l[2]*f*u,this.target[1]=Math.max(2,this.target[1]+c*u)}});const i=()=>t=0;e.addEventListener("pointerup",i),e.addEventListener("pointercancel",i),e.addEventListener("wheel",n=>{n.preventDefault(),this.dist=Math.min(900,Math.max(40,this.dist*Math.exp(n.deltaY*.001)))},{passive:!1}),e.addEventListener("contextmenu",n=>n.preventDefault())}viewProj(e){const t=Math.cos(this.phi),r=Math.sin(this.phi),o=[this.target[0]+this.dist*t*Math.sin(this.theta),this.target[1]+this.dist*r,this.target[2]+this.dist*t*Math.cos(this.theta)],i=ze(o,this.target,[0,1,0]),n=Ue(50*Math.PI/180,e,1,3e3);return de(n,i)}}const b=s=>document.getElementById(s);class Ne{constructor(e){g(this,"paused",!1);this.h=e;const t=(r,o,i,n)=>{const f=b(r),c=b(o),u=()=>{const l=parseFloat(f.value);c.textContent=i(l),n(l)};f.addEventListener("input",u),u()};t("s-wind","v-wind",r=>r.toFixed(3),r=>e.onWind(r)),t("s-nu","v-nu",r=>r.toFixed(4),r=>e.onViscosity(r)),t("s-slice","v-slice",r=>(r*100).toFixed(0)+"%",r=>e.onSlicePos(r)),t("s-mirror","v-mirror",r=>(r*100).toFixed(0)+"%",()=>e.onCarParam()),t("s-spoiler","v-spoiler",r=>r.toFixed(0)+"°",()=>e.onCarParam()),t("s-ride","v-ride",r=>(r>=0?"+":"")+(r*100).toFixed(0)+"cm",()=>e.onCarParam()),b("c-smoke").addEventListener("change",r=>e.onSmoke(r.target.checked)),b("c-slice").addEventListener("change",r=>e.onSlice(r.target.checked)),b("c-spoiler").addEventListener("change",()=>e.onCarParam()),b("btn-pause").addEventListener("click",()=>{this.paused=!this.paused,b("btn-pause").textContent=this.paused?"继续":"暂停",e.onPause(this.paused)}),b("btn-reset").addEventListener("click",()=>e.onReset()),b("btn-a").addEventListener("click",()=>e.onSaveA()),b("btn-b").addEventListener("click",()=>e.onSaveB()),b("btn-flip").addEventListener("click",()=>e.onFlip()),b("btn-default").addEventListener("click",()=>e.onRestoreDefault()),b("btn-load").addEventListener("click",()=>b("file").click()),b("file").addEventListener("change",r=>{var i;const o=(i=r.target.files)==null?void 0:i[0];o&&e.onLoadFile(o),r.target.value=""})}getCarParams(){return{mirrorPos:parseFloat(b("s-mirror").value),spoilerAngle:parseFloat(b("s-spoiler").value),spoilerOn:b("c-spoiler").checked,rideHeight:parseFloat(b("s-ride").value)}}setProceduralEnabled(e){for(const t of document.querySelectorAll(".proc"))t.classList.toggle("disabled",!e)}setStats(e){b("st-grid").textContent=e.grid,b("st-steps").textContent=e.stepsPerSec,b("st-re").textContent=e.re,b("st-area").textContent=e.area}setCd(e,t){b("st-cd").textContent=e===null?"—":e.toFixed(3),b("st-cl").textContent=t===null?"—":t.toFixed(3)}setCompare(e,t){b("cd-a").textContent=e===null?"—":e.toFixed(3),b("cd-b").textContent=t===null?"—":t.toFixed(3);const r=b("cd-delta");if(e!==null&&t!==null){const o=t-e;r.textContent=`${o>=0?"+":""}${o.toFixed(3)} (${(o/e*100).toFixed(1)}%)`,r.className=o>=0?"pos":"neg",r.id="cd-delta"}else r.textContent="—",r.className=""}toast(e,t=2600){const r=b("toast");r.textContent=e,r.style.opacity="1",setTimeout(()=>r.style.opacity="0",t)}}function Le(s){const e=b("fatal");e.textContent=s,e.style.display="flex"}class _e{constructor(){g(this,"pos",[]);g(this,"idx",[])}vert(e,t,r){return this.pos.push(e,t,r),this.pos.length/3-1}tri(e,t,r){this.idx.push(e,t,r)}quad(e,t,r,o){this.tri(e,t,r),this.tri(e,r,o)}loft(e,t=14){const r=[];for(const o of e){const i=(o.yBot+o.yTop)/2,n=(o.yTop-o.yBot)/2,f=[];for(let c=0;c<t;c++){const u=c/t*Math.PI*2;f.push(this.vert(o.x,i+n*Math.sin(u),o.w*Math.cos(u)))}r.push(f)}for(let o=0;o<r.length-1;o++)for(let i=0;i<t;i++){const n=(i+1)%t;this.quad(r[o][i],r[o][n],r[o+1][n],r[o+1][i])}for(const[o,i,n]of[[r[0],e[0].x,!0],[r[r.length-1],e[e.length-1].x,!1]]){const f=o.reduce((u,l)=>u+this.pos[l*3+1],0)/o.length,c=this.vert(i,f,0);for(let u=0;u<t;u++){const l=(u+1)%t;n?this.tri(c,o[l],o[u]):this.tri(c,o[u],o[l])}}}box(e,t,r,o,i,n,f=0){const c=Math.cos(f),u=Math.sin(f),l=[];for(const[a,m,v]of[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]){const p=a*o/2,h=m*i/2;l.push(this.vert(e+p*c-h*u,t+p*u+h*c,r+v*n/2))}const d=[[0,1,2,3],[5,4,7,6],[4,0,3,7],[1,5,6,2],[3,2,6,7],[4,5,1,0]];for(const a of d)this.quad(l[a[0]],l[a[1]],l[a[2]],l[a[3]])}cylinder(e,t,r,o,i,n=14){const f=[],c=[];for(let d=0;d<n;d++){const a=d/n*Math.PI*2;f.push(this.vert(e+o*Math.cos(a),t+o*Math.sin(a),r-i/2)),c.push(this.vert(e+o*Math.cos(a),t+o*Math.sin(a),r+i/2))}for(let d=0;d<n;d++){const a=(d+1)%n;this.quad(f[d],f[a],c[a],c[d])}const u=this.vert(e,t,r-i/2),l=this.vert(e,t,r+i/2);for(let d=0;d<n;d++){const a=(d+1)%n;this.tri(u,f[a],f[d]),this.tri(l,c[d],c[a])}}build(){const e=new Float32Array(this.pos),t=new Uint32Array(this.idx),r=q(e,t);return{positions:e,normals:r,indices:t}}}function q(s,e){const t=s.length/3,r=new Map,o=new Uint32Array(t),i=[];for(let c=0;c<t;c++){const u=`${s[c*3].toFixed(4)},${s[c*3+1].toFixed(4)},${s[c*3+2].toFixed(4)}`;let l=r.get(u);l===void 0&&(l=i.length/3,r.set(u,l),i.push(s[c*3],s[c*3+1],s[c*3+2])),o[c]=l}const n=new Float32Array(i.length);for(let c=0;c<e.length;c+=3){const u=o[e[c]],l=o[e[c+1]],d=o[e[c+2]],a=i[u*3],m=i[u*3+1],v=i[u*3+2],p=i[l*3]-a,h=i[l*3+1]-m,B=i[l*3+2]-v,y=i[d*3]-a,P=i[d*3+1]-m,S=i[d*3+2]-v,w=h*S-B*P,A=B*y-p*S,M=p*P-h*y;for(const T of[u,l,d])n[T*3]+=w,n[T*3+1]+=A,n[T*3+2]+=M}const f=new Float32Array(t*3);for(let c=0;c<t;c++){const u=o[c],l=Math.hypot(n[u*3],n[u*3+1],n[u*3+2])||1;f[c*3]=n[u*3]/l,f[c*3+1]=n[u*3+1]/l,f[c*3+2]=n[u*3+2]/l}return f}function Ee(s){const e=new _e,t=s.rideHeight;e.loft([{x:-2.2,w:.55,yBot:.35+t,yTop:.95+t},{x:-1.9,w:.8,yBot:.28+t,yTop:1.05+t},{x:-1.5,w:.88,yBot:.22+t,yTop:1.1+t},{x:-.8,w:.9,yBot:.18+t,yTop:1.12+t},{x:0,w:.9,yBot:.16+t,yTop:1.1+t},{x:.8,w:.88,yBot:.18+t,yTop:.95+t},{x:1.4,w:.84,yBot:.22+t,yTop:.8+t},{x:1.9,w:.75,yBot:.28+t,yTop:.68+t},{x:2.2,w:.6,yBot:.35+t,yTop:.6+t}]),e.loft([{x:-1.6,w:.55,yBot:1+t,yTop:1.1+t},{x:-1.2,w:.72,yBot:1.05+t,yTop:1.35+t},{x:-.5,w:.78,yBot:1.08+t,yTop:1.45+t},{x:.2,w:.76,yBot:1.05+t,yTop:1.44+t},{x:.8,w:.68,yBot:.95+t,yTop:1.25+t},{x:.95,w:.6,yBot:.9+t,yTop:1.05+t}]);for(const o of[-1.45,1.45])for(const i of[-.72,.72])e.cylinder(o,.33,i,.33,.22);const r=.5+s.mirrorPos*.5;for(const o of[-1,1]){const i=o*1;e.box(r,.98+t,o*.93,.04,.03,.16),e.box(r,1+t,i,.07,.09,.16)}if(s.spoilerOn){const o=-s.spoilerAngle*Math.PI/180;for(const i of[-1,1])e.box(-1.98,1.05+t,i*.42,.06,.28,.05);e.box(-2.05,1.2+t,0,.32,.025,1.35,o)}return e.build()}function Re(s,e){const t=s.positions;let r=1e9,o=-1e9,i=1e9,n=-1e9,f=1e9,c=-1e9;for(let y=0;y<t.length;y+=3)r=Math.min(r,t[y]),o=Math.max(o,t[y]),i=Math.min(i,t[y+1]),n=Math.max(n,t[y+1]),f=Math.min(f,t[y+2]),c=Math.max(c,t[y+2]);const u=o-r,l=c-f,d=l>u,m=4.4/Math.max(u,l),v=(r+o)/2,p=(f+c)/2,h=new Float32Array(t.length);for(let y=0;y<t.length;y+=3){let P=(t[y]-v)*m;const S=(t[y+1]-i)*m;let w=(t[y+2]-p)*m;d&&([P,w]=[w,-P]),e&&(P=-P),h[y]=P,h[y+1]=S,h[y+2]=w}const B=q(h,s.indices);return{positions:h,normals:B,indices:s.indices}}function $e(s,e,t,r,o,i=!0){const n=s.positions.length,f=new Float32Array(n),c=i?-e:e;for(let u=0;u<n;u+=3)f[u]=s.positions[u]*c+t,f[u+1]=s.positions[u+1]*e+r,f[u+2]=s.positions[u+2]*e+o;return{positions:f,normals:s.normals,indices:s.indices}}async function De(s){const e=await s.arrayBuffer();if(/\.stl$/i.test(s.name))return Ve(e);if(/\.glb$/i.test(s.name))return Ye(e);throw new Error("仅支持 .glb / .stl 文件")}function Ve(s){const e=new Uint8Array(s),t=new TextDecoder().decode(e.slice(0,512)),r=t.trimStart().startsWith("solid")&&t.includes("facet"),o=[];if(r){const c=new TextDecoder().decode(e),u=/vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)/g;let l;for(;l=u.exec(c);)o.push(+l[1],+l[2],+l[3])}else{const c=new DataView(s),u=c.getUint32(80,!0);for(let l=0;l<u;l++){const d=84+l*50+12;for(let a=0;a<3;a++)o.push(c.getFloat32(d+a*12,!0),c.getFloat32(d+a*12+4,!0),c.getFloat32(d+a*12+8,!0))}}const i=new Float32Array(o),n=new Uint32Array(i.length/3);for(let c=0;c<n.length;c++)n[c]=c;const f=q(i,n);return{positions:i,normals:f,indices:n}}function Ye(s){const e=new DataView(s);if(e.getUint32(0,!0)!==1179937895)throw new Error("不是有效的 GLB 文件");const t=e.getUint32(12,!0),r=JSON.parse(new TextDecoder().decode(new Uint8Array(s,20,t))),o=20+t+8,i=m=>{const v=r.accessors[m],p=r.bufferViews[v.bufferView],h=o+(p.byteOffset??0)+(v.byteOffset??0),B={SCALAR:1,VEC2:2,VEC3:3,VEC4:4}[v.type]??3,y=p.byteStride??0;let P;if(v.componentType===5126)if(P=new Float32Array(v.count*B),y===B*4||y===0)P.set(new Float32Array(s,h,v.count*B));else for(let S=0;S<v.count;S++)for(let w=0;w<B;w++)P[S*B+w]=e.getFloat32(h+S*y+w*4,!0);else if(v.componentType===5125)P=new Uint32Array(s.slice(h,h+v.count*4));else if(v.componentType===5123)P=new Uint16Array(s.slice(h,h+v.count*2));else if(v.componentType===5121)P=new Uint16Array(new Uint8Array(s.slice(h,h+v.count)));else throw new Error(`不支持的 accessor 类型 ${v.componentType}`);return{data:P,count:v.count,comps:B}},n=[],f=[],c=(m,v)=>{const p=r.nodes[m];let h;p.matrix?h=new Float32Array(p.matrix):h=Te(p.translation??[0,0,0],p.rotation??[0,0,0,1],p.scale??[1,1,1]);const B=de(v,h);if(p.mesh!==void 0)for(const y of r.meshes[p.mesh].primitives){if((y.mode??4)!==4||y.attributes.POSITION===void 0)continue;const P=i(y.attributes.POSITION),S=n.length/3;for(let w=0;w<P.count;w++){const A=Ie(B,P.data[w*3],P.data[w*3+1],P.data[w*3+2]);n.push(A[0],A[1],A[2])}if(y.indices!==void 0){const w=i(y.indices);for(let A=0;A<w.count;A++)f.push(S+w.data[A])}else for(let w=0;w<P.count;w++)f.push(S+w)}for(const y of p.children??[])c(y,B)},u=r.scenes[r.scene??0];for(const m of u.nodes)c(m,j());if(n.length===0)throw new Error("GLB 中没有可用的三角网格");const l=new Float32Array(n),d=new Uint32Array(f),a=q(l,d);return{positions:l,normals:a,indices:d}}function ke(s,e,t,r,o){const i=[o[0]-s,o[1]-e,o[2]-t,o[3]-s,o[4]-e,o[5]-t,o[6]-s,o[7]-e,o[8]-t],n=[i[3]-i[0],i[4]-i[1],i[5]-i[2],i[6]-i[3],i[7]-i[4],i[8]-i[5],i[0]-i[6],i[1]-i[7],i[2]-i[8]],f=(d,a,m)=>{const v=d*i[0]+a*i[1]+m*i[2],p=d*i[3]+a*i[4]+m*i[5],h=d*i[6]+a*i[7]+m*i[8],B=r*(Math.abs(d)+Math.abs(a)+Math.abs(m)),y=Math.min(v,p,h),P=Math.max(v,p,h);return y>B||P<-B};for(let d=0;d<3;d++){const a=n[d*3],m=n[d*3+1],v=n[d*3+2];if(f(0,-v,m)||f(v,0,-a)||f(-m,a,0))return!1}for(let d=0;d<3;d++){const a=Math.min(i[d],i[3+d],i[6+d]),m=Math.max(i[d],i[3+d],i[6+d]);if(a>r||m<-r)return!1}const c=n[1]*n[5]-n[2]*n[4],u=n[2]*n[3]-n[0]*n[5],l=n[0]*n[4]-n[1]*n[3];return!f(c,u,l)}function Xe(s,e,t,r){const o=e*t*r,i=new Uint8Array(o),n=s.positions,f=s.indices,c=new Array(9);for(let a=0;a<f.length;a+=3){for(let p=0;p<3;p++){const h=f[a+p]*3;c[p*3]=n[h],c[p*3+1]=n[h+1],c[p*3+2]=n[h+2]}const m=[0,0,0],v=[0,0,0];for(let p=0;p<3;p++){const h=Math.min(c[p],c[3+p],c[6+p]),B=Math.max(c[p],c[3+p],c[6+p]);m[p]=Math.max(1,Math.floor(h-.5)),v[p]=Math.min(e-2,Math.ceil(B+.5))}v[1]=Math.min(t-2,v[1]),v[2]=Math.min(r-2,v[2]),m[1]=Math.max(1,m[1]),m[2]=Math.max(1,m[2]);for(let p=m[2];p<=v[2];p++)for(let h=m[1];h<=v[1];h++)for(let B=m[0];B<=v[0];B++){const y=B+h*e+p*e*t;i[y]===0&&ke(B+.5,h+.5,p+.5,.5,c)&&(i[y]=1)}}const u=[],l=a=>{i[a]===0&&(i[a]=2,u.push(a))};for(let a=0;a<r;a++)for(let m=0;m<t;m++)for(const v of[0,e-1])l(v+m*e+a*e*t);for(let a=0;a<r;a++)for(let m=0;m<e;m++)for(const v of[0,t-1])l(m+v*e+a*e*t);for(let a=0;a<t;a++)for(let m=0;m<e;m++)for(const v of[0,r-1])l(m+a*e+v*e*t);for(;u.length;){const a=u.pop(),m=a%e,v=(a/e|0)%t,p=a/(e*t)|0;m>0&&l(a-1),m<e-1&&l(a+1),v>0&&l(a-e),v<t-1&&l(a+e),p>0&&l(a-e*t),p<r-1&&l(a+e*t)}for(let a=0;a<o;a++)i[a]===0?i[a]=1:i[a]===2&&(i[a]=0);let d=0;for(let a=0;a<r;a++)for(let m=0;m<t;m++){let v=!1;const p=m*e+a*e*t;for(let h=0;h<e;h++)if(i[p+h]===1){v=!0;break}v&&d++}return{mask:i,area:d}}const fe=[[256,96,96],[224,96,96],[256,88,96]],X=25,le=100,qe=1.5;async function He(){const s=document.getElementById("gpu");let e=null,t=fe[0],r=null;for(const x of fe){const G=x[0]*x[1]*x[2]*19*4;try{e=await ve(G),t=x;break}catch(I){r=I}}if(!e)throw r;const[o,i,n]=t,f=s.getContext("webgpu"),c=navigator.gpu.getPreferredCanvasFormat();f.configure({device:e,format:c,alphaMode:"opaque"});const u=new Se(e,o,i,n),l=e.createBuffer({size:80,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),d=new Oe(s);d.target=[le,18,n/2];const a=new Ae(e,o,i,n,l,u.macroBuf,u.flagsBuf,c),m=new Ge(e,o,i,n,u.macroBuf,u.flagsBuf,l,c);let v=1,p=!0,h=null,B=!1,y=null;const P=(x,G)=>{const I=$e(x,X,le,qe,n/2,G),{mask:F,area:H}=Xe(I,o,i,n);v=Math.max(1,H),u.updateObstacles(F),a.setMesh(I)},S=()=>{h&&P(Re(h,B),!1)},w=()=>{P(Ee(C.getCarParams()),!0)},A=()=>{y&&clearTimeout(y),y=setTimeout(()=>{y=null,p&&w()},150)};let M=!1,T=!0,J=!0,K=.5,D=null,V=null,U=null,N=null;const C=new Ne({onWind:x=>u.setFlow(x,u.nu),onViscosity:x=>u.setFlow(u.u0,x),onSmoke:x=>T=x,onSlice:x=>J=x,onSlicePos:x=>K=x,onPause:x=>M=x,onReset:()=>{u.resetFlow(),U=N=null,C.toast("流场已重置为均匀来流")},onCarParam:()=>A(),onSaveA:()=>{if(U===null)return C.toast("Cd 还没有读数,等流场发展一下");D=U,C.setCompare(D,V),C.toast(`方案 A:Cd = ${D.toFixed(3)}`)},onSaveB:()=>{if(U===null)return C.toast("Cd 还没有读数,等流场发展一下");V=U,C.setCompare(D,V),C.toast(`方案 B:Cd = ${V.toFixed(3)}`)},onLoadFile:async x=>{try{C.toast(`正在解析 ${x.name}…`),h=await De(x),p=!1,C.setProceduralEnabled(!1),C.toast(`体素化中(${(h.indices.length/3).toLocaleString()} 三角形)…`),await new Promise(G=>setTimeout(G,30)),S(),C.toast("模型已加载,流场继续推进")}catch(G){C.toast(`加载失败:${G.message}`,5e3)}},onFlip:()=>{B=!B,p||S(),C.toast("模型已掉头")},onRestoreDefault:()=>{p=!0,h=null,C.setProceduralEnabled(!0),w(),C.toast("已恢复内置参数化车型")}});w(),m.prime();let L=null;const pe=()=>{const x=Math.max(1,Math.floor(s.clientWidth*devicePixelRatio)),G=Math.max(1,Math.floor(s.clientHeight*devicePixelRatio));(s.width!==x||s.height!==G)&&(s.width=x,s.height=G,L==null||L.destroy(),L=e.createTexture({size:[x,G],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT}))};let z=4,_=16.7,Q=performance.now(),ee=0,te=0;const he=4.4*X,re=()=>{const x=performance.now(),G=x-Q;Q=x,_=_*.9+G*.1,M||(_>24&&z>1?z--:_<19&&z<64&&z++),te=M?0:z*1e3/_,pe();const I=s.width/s.height;e.queue.writeBuffer(l,0,d.viewProj(I));const F=e.createCommandEncoder();M||(u.encodeSteps(F,z),m.encodeAdvect(F,z),a.encodeCarColors(F,u.u0));const H=f.getCurrentTexture().createView(),E=F.beginRenderPass({colorAttachments:[{view:H,clearValue:{r:.045,g:.055,b:.075,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:L.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});if(a.encodeGround(E),J&&(a.setSlice(K,u.u0),a.encodeSlice(E)),a.encodeCar(E),T&&m.encodeRender(E),E.end(),e.queue.submit([F.finish()]),M||u.readForce().then(R=>{if(!R)return;const Y=.5*u.u0*u.u0*v,ie=R.fx/Y,oe=R.fy/Y;U=U===null?ie:U*.92+ie*.08,N=N===null?oe:N*.92+oe*.08,C.setCd(U,N)}),x-ee>300){ee=x;const R=u.nu,Y=u.u0*he/R;C.setStats({grid:`${o}×${i}×${n} = ${(o*i*n/1e6).toFixed(2)}M`,stepsPerSec:te.toFixed(0),re:Y.toExponential(2),area:`${v} 格² (≈${(v/(X*X)).toFixed(2)} m²)`})}requestAnimationFrame(re)};requestAnimationFrame(re)}He().catch(s=>{console.error(s),Le(`初始化失败

${s.message??s}`)});
