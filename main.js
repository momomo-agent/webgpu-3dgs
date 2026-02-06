// WebGPU/WebGL 3D Gaussian Splatting Viewer with Fallback
const canvas = document.getElementById('canvas');
const status = document.getElementById('status');
const fpsEl = document.getElementById('fps');
const loadBtn = document.getElementById('loadBtn');
const fileInput = document.getElementById('fileInput');

let renderer = null; // 'webgpu' or 'webgl'
let device, context, format; // WebGPU
let gl; // WebGL
let camera = { x: 0, y: 0, z: 5, rotX: 0.5, rotY: 0 };
let isDragging = false;
let lastMouse = { x: 0, y: 0 };
let animating = false;

// Shared state
let vertexBuffer = null;
let uniformBuffer = null;
let bindGroup = null;
let pipeline = null;
let vertexCount = 0;
let depthTexture = null;
let frameCount = 0;
let fpsUpdateTime = 0;
let pointSizeMultiplier = 5;

// WebGL specific
let glProgram = null;
let glVertexBuffer = null;
let glVertexData = null;

// ============ INIT ============
async function init() {
  // Try WebGPU first
  if (await initWebGPU()) {
    renderer = 'webgpu';
    status.textContent = 'WebGPU ✓';
    createWebGPUPipeline();
  } else if (initWebGL()) {
    renderer = 'webgl';
    status.textContent = 'WebGL ✓';
    createWebGLProgram();
  } else {
    status.textContent = 'No WebGPU/WebGL support';
    status.style.color = '#f00';
    return;
  }
  
  const { data, count } = generateDefaultCloud();
  loadCloud(data, count);
  requestAnimationFrame(render);
}

// ============ WebGPU ============
async function initWebGPU() {
  if (!navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return false;
    device = await adapter.requestDevice();
    context = canvas.getContext('webgpu');
    if (!context) return false;
    format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    resizeWebGPU();
    return true;
  } catch (e) {
    console.warn('WebGPU init failed:', e);
    return false;
  }
}

function resizeWebGPU() {
  canvas.width = window.innerWidth * devicePixelRatio;
  canvas.height = window.innerHeight * devicePixelRatio;
  if (device) {
    depthTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
  }
}

const webgpuShader = `
struct Uniforms { mvp: mat4x4f, pointSize: f32, aspect: f32 }
@group(0) @binding(0) var<uniform> u: Uniforms;
struct VSOut { @builtin(position) pos: vec4f, @location(0) col: vec4f, @location(1) uv: vec2f }

@vertex fn vs(@location(0) p: vec3f, @location(1) c: vec4f, @builtin(vertex_index) vid: u32) -> VSOut {
  var o: VSOut;
  let basePos = u.mvp * vec4f(p, 1.0);
  let quadIdx = vid % 6u;
  let size = u.pointSize * 0.001;
  var offset = vec2f(0.0);
  var uv = vec2f(0.0);
  if (quadIdx == 0u) { offset = vec2f(-1.0, -1.0); uv = vec2f(-1.0, -1.0); }
  else if (quadIdx == 1u) { offset = vec2f(1.0, -1.0); uv = vec2f(1.0, -1.0); }
  else if (quadIdx == 2u) { offset = vec2f(1.0, 1.0); uv = vec2f(1.0, 1.0); }
  else if (quadIdx == 3u) { offset = vec2f(-1.0, -1.0); uv = vec2f(-1.0, -1.0); }
  else if (quadIdx == 4u) { offset = vec2f(1.0, 1.0); uv = vec2f(1.0, 1.0); }
  else { offset = vec2f(-1.0, 1.0); uv = vec2f(-1.0, 1.0); }
  o.pos = basePos + vec4f(offset.x * size / u.aspect, offset.y * size, 0.0, 0.0);
  o.col = c;
  o.uv = uv;
  return o;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  let alpha = exp(-d * d * 2.0);
  return vec4f(in.col.rgb, in.col.a * alpha);
}
`;

function createWebGPUPipeline() {
  const shader = device.createShaderModule({ code: webgpuShader });
  uniformBuffer = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const layout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} }]
  });
  bindGroup = device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: uniformBuffer } }] });
  pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    vertex: {
      module: shader, entryPoint: 'vs',
      buffers: [{ arrayStride: 28, attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x4' }
      ]}]
    },
    fragment: { module: shader, entryPoint: 'fs', targets: [{ 
      format,
      blend: {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
      }
    }] },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' }
  });
}

// ============ WebGL Fallback ============
function initWebGL() {
  try {
    gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) return false;
    resizeWebGL();
    return true;
  } catch (e) {
    return false;
  }
}

function resizeWebGL() {
  canvas.width = window.innerWidth * devicePixelRatio;
  canvas.height = window.innerHeight * devicePixelRatio;
  if (gl) gl.viewport(0, 0, canvas.width, canvas.height);
}

const webglVS = `
attribute vec3 aPosition;
attribute vec4 aColor;
attribute vec2 aOffset;
uniform mat4 uMVP;
uniform float uPointSize;
uniform float uAspect;
varying vec4 vColor;
varying vec2 vUV;
void main() {
  vec4 basePos = uMVP * vec4(aPosition, 1.0);
  float size = uPointSize * 0.001;
  gl_Position = basePos + vec4(aOffset.x * size / uAspect, aOffset.y * size, 0.0, 0.0);
  vColor = aColor;
  vUV = aOffset;
}
`;

const webglFS = `
precision mediump float;
varying vec4 vColor;
varying vec2 vUV;
void main() {
  float d = length(vUV);
  if (d > 1.0) discard;
  float alpha = exp(-d * d * 2.0);
  gl_FragColor = vec4(vColor.rgb, vColor.a * alpha);
}
`;

function createWebGLProgram() {
  const vs = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vs, webglVS);
  gl.compileShader(vs);
  
  const fs = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fs, webglFS);
  gl.compileShader(fs);
  
  glProgram = gl.createProgram();
  gl.attachShader(glProgram, vs);
  gl.attachShader(glProgram, fs);
  gl.linkProgram(glProgram);
  gl.useProgram(glProgram);
  
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.enable(gl.DEPTH_TEST);
}

// ============ Shared Math ============
function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)/(near-far),-1, 0,0,2*far*near/(near-far),0]);
}

function lookAt() {
  const cx = Math.cos(camera.rotX), sx = Math.sin(camera.rotX);
  const cy = Math.cos(camera.rotY), sy = Math.sin(camera.rotY);
  return new Float32Array([cy,sx*sy,-cx*sy,0, 0,cx,sx,0, sy,-sx*cy,cx*cy,0, 0,0,-camera.z,1]);
}

function mulMat4(a, b) {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++)
    o[i*4+j] = a[i*4]*b[j] + a[i*4+1]*b[4+j] + a[i*4+2]*b[8+j] + a[i*4+3]*b[12+j];
  return o;
}

// ============ Scene Generation ============
function generateScene() {
  const pts = [];
  const add = (cx,cy,cz,rx,ry,rz,n,col) => {
    for (let i = 0; i < n; i++) {
      const u = Math.random(), v = Math.random();
      const theta = 2 * Math.PI * u, phi = Math.acos(2 * v - 1);
      const vary = 0.9 + Math.random() * 0.2;
      pts.push({
        x: cx + rx * Math.sin(phi) * Math.cos(theta),
        y: cy + ry * Math.cos(phi),
        z: cz + rz * Math.sin(phi) * Math.sin(theta),
        r: Math.min(1, col[0] * vary), 
        g: Math.min(1, col[1] * vary), 
        b: Math.min(1, col[2] * vary)
      });
    }
  };
  
  // Bunny
  add(0, 0, 0, 0.5, 0.4, 0.6, 10000, [0.95, 0.9, 0.85]);
  add(0, 0.45, 0.4, 0.3, 0.3, 0.32, 6000, [0.97, 0.92, 0.87]);
  add(-0.12, 0.85, 0.35, 0.07, 0.28, 0.05, 2500, [1, 0.85, 0.9]);
  add(0.12, 0.85, 0.35, 0.07, 0.28, 0.05, 2500, [1, 0.85, 0.9]);
  add(0, 0.12, -0.6, 0.14, 0.14, 0.12, 2000, [1, 0.98, 0.95]);
  add(-0.18, -0.4, 0.25, 0.1, 0.18, 0.1, 1500, [0.92, 0.87, 0.82]);
  add(0.18, -0.4, 0.25, 0.1, 0.18, 0.1, 1500, [0.92, 0.87, 0.82]);
  add(-0.22, -0.35, -0.3, 0.14, 0.2, 0.18, 2000, [0.92, 0.87, 0.82]);
  add(0.22, -0.35, -0.3, 0.14, 0.2, 0.18, 2000, [0.92, 0.87, 0.82]);
  add(-0.1, 0.5, 0.65, 0.04, 0.04, 0.02, 500, [0.1, 0.1, 0.1]);
  add(0.1, 0.5, 0.65, 0.04, 0.04, 0.02, 500, [0.1, 0.1, 0.1]);
  add(0, 0.38, 0.7, 0.03, 0.02, 0.02, 300, [1, 0.6, 0.7]);
  
  return pts;
}

function generateDefaultCloud() {
  const points = generateScene();
  const count = points.length * 6;
  const data = new Float32Array(count * 7);
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    for (let j = 0; j < 6; j++) {
      const idx = (i * 6 + j) * 7;
      data[idx] = p.x; data[idx+1] = p.y; data[idx+2] = p.z;
      data[idx+3] = p.r; data[idx+4] = p.g; data[idx+5] = p.b; data[idx+6] = 1.0;
    }
  }
  return { data, count };
}

// ============ Load Cloud ============
function loadCloud(data, count) {
  vertexCount = count;
  
  if (renderer === 'webgpu') {
    vertexBuffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(vertexBuffer, 0, data);
  } else {
    // WebGL: expand data to include offset attribute
    const pointCount = count / 6;
    const offsets = [[-1,-1], [1,-1], [1,1], [-1,-1], [1,1], [-1,1]];
    glVertexData = new Float32Array(count * 9); // pos(3) + color(4) + offset(2)
    
    for (let i = 0; i < pointCount; i++) {
      for (let j = 0; j < 6; j++) {
        const srcIdx = (i * 6 + j) * 7;
        const dstIdx = (i * 6 + j) * 9;
        glVertexData[dstIdx] = data[srcIdx];
        glVertexData[dstIdx+1] = data[srcIdx+1];
        glVertexData[dstIdx+2] = data[srcIdx+2];
        glVertexData[dstIdx+3] = data[srcIdx+3];
        glVertexData[dstIdx+4] = data[srcIdx+4];
        glVertexData[dstIdx+5] = data[srcIdx+5];
        glVertexData[dstIdx+6] = data[srcIdx+6];
        glVertexData[dstIdx+7] = offsets[j][0];
        glVertexData[dstIdx+8] = offsets[j][1];
      }
    }
    
    if (!glVertexBuffer) glVertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, glVertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, glVertexData, gl.STATIC_DRAW);
  }
  
  status.textContent = `${(count/6).toLocaleString()} pts (${renderer.toUpperCase()})`;
}

// ============ Render ============
function render(time) {
  if (animating && !isDragging) camera.rotY += 0.005;
  frameCount++;
  if (time - fpsUpdateTime >= 1000) { 
    fpsEl.textContent = `${frameCount} FPS`; 
    frameCount = 0; 
    fpsUpdateTime = time; 
  }
  
  if (renderer === 'webgpu') renderWebGPU();
  else renderWebGL();
  
  requestAnimationFrame(render);
}

function renderWebGPU() {
  if (!pipeline || !vertexBuffer) return;
  
  const aspect = canvas.width / canvas.height;
  const mvp = mulMat4(perspective(Math.PI / 4, aspect, 0.1, 100), lookAt());
  const uniformData = new Float32Array(20);
  uniformData.set(mvp, 0); 
  uniformData[16] = pointSizeMultiplier; 
  uniformData[17] = aspect;
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);
  
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{ 
      view: context.getCurrentTexture().createView(), 
      clearValue: { r: 0.02, g: 0.02, b: 0.03, a: 1 }, 
      loadOp: 'clear', 
      storeOp: 'store' 
    }],
    depthStencilAttachment: { 
      view: depthTexture.createView(), 
      depthClearValue: 1.0, 
      depthLoadOp: 'clear', 
      depthStoreOp: 'store' 
    }
  });
  pass.setPipeline(pipeline); 
  pass.setBindGroup(0, bindGroup); 
  pass.setVertexBuffer(0, vertexBuffer);
  pass.draw(vertexCount); 
  pass.end();
  device.queue.submit([encoder.finish()]);
}

function renderWebGL() {
  if (!glProgram || !glVertexBuffer) return;
  
  gl.clearColor(0.02, 0.02, 0.03, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  
  const aspect = canvas.width / canvas.height;
  const mvp = mulMat4(perspective(Math.PI / 4, aspect, 0.1, 100), lookAt());
  
  gl.useProgram(glProgram);
  gl.uniformMatrix4fv(gl.getUniformLocation(glProgram, 'uMVP'), false, mvp);
  gl.uniform1f(gl.getUniformLocation(glProgram, 'uPointSize'), pointSizeMultiplier);
  gl.uniform1f(gl.getUniformLocation(glProgram, 'uAspect'), aspect);
  
  gl.bindBuffer(gl.ARRAY_BUFFER, glVertexBuffer);
  const stride = 36; // 9 floats * 4 bytes
  
  const aPos = gl.getAttribLocation(glProgram, 'aPosition');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, stride, 0);
  
  const aCol = gl.getAttribLocation(glProgram, 'aColor');
  gl.enableVertexAttribArray(aCol);
  gl.vertexAttribPointer(aCol, 4, gl.FLOAT, false, stride, 12);
  
  const aOff = gl.getAttribLocation(glProgram, 'aOffset');
  gl.enableVertexAttribArray(aOff);
  gl.vertexAttribPointer(aOff, 2, gl.FLOAT, false, stride, 28);
  
  gl.drawArrays(gl.TRIANGLES, 0, vertexCount);
}

// ============ Resize ============
function resize() {
  if (renderer === 'webgpu') resizeWebGPU();
  else if (renderer === 'webgl') resizeWebGL();
}

// ============ Events ============
canvas.addEventListener('mousedown', e => { isDragging = true; lastMouse = { x: e.clientX, y: e.clientY }; });
canvas.addEventListener('mousemove', e => {
  if (!isDragging) return;
  camera.rotY += (e.clientX - lastMouse.x) * 0.01;
  camera.rotX += (e.clientY - lastMouse.y) * 0.01;
  camera.rotX = Math.max(-1.5, Math.min(1.5, camera.rotX));
  lastMouse = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener('mouseup', () => isDragging = false);
canvas.addEventListener('mouseleave', () => isDragging = false);
canvas.addEventListener('wheel', e => { 
  camera.z += e.deltaY * 0.005; 
  camera.z = Math.max(1, Math.min(10, camera.z)); 
});

// Touch support for mobile
canvas.addEventListener('touchstart', e => {
  if (e.touches.length === 1) {
    isDragging = true;
    lastMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
  }
});
canvas.addEventListener('touchmove', e => {
  if (!isDragging || e.touches.length !== 1) return;
  const touch = e.touches[0];
  camera.rotY += (touch.clientX - lastMouse.x) * 0.01;
  camera.rotX += (touch.clientY - lastMouse.y) * 0.01;
  camera.rotX = Math.max(-1.5, Math.min(1.5, camera.rotX));
  lastMouse = { x: touch.clientX, y: touch.clientY };
  e.preventDefault();
}, { passive: false });
canvas.addEventListener('touchend', () => isDragging = false);

loadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async e => {
  const file = e.target.files[0]; if (!file) return;
  status.textContent = 'Loading...';
  const buffer = await file.arrayBuffer();
  await parsePLY(buffer);
});

async function parsePLY(buffer) {
  const text = new TextDecoder().decode(buffer);
  const lines = text.split('\n');
  let pointCount = 0, headerEnd = 0;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('element vertex')) pointCount = parseInt(lines[i].split(' ')[2]);
    if (lines[i] === 'end_header') { headerEnd = i + 1; break; }
  }
  const vertCount = pointCount * 6;
  const data = new Float32Array(vertCount * 7);
  for (let i = 0; i < pointCount; i++) {
    const p = lines[headerEnd + i].trim().split(/\s+/);
    const x = parseFloat(p[0]), y = parseFloat(p[1]), z = parseFloat(p[2]);
    const r = (parseFloat(p[3]) || 128) / 255;
    const g = (parseFloat(p[4]) || 128) / 255;
    const b = (parseFloat(p[5]) || 128) / 255;
    for (let j = 0; j < 6; j++) {
      const idx = (i * 6 + j) * 7;
      data[idx] = x; data[idx+1] = y; data[idx+2] = z;
      data[idx+3] = r; data[idx+4] = g; data[idx+5] = b; data[idx+6] = 1.0;
    }
  }
  loadCloud(data, vertCount);
}

window.addEventListener('resize', resize);

// Point size slider
const sizeSlider = document.getElementById('sizeSlider');
const sizeVal = document.getElementById('sizeVal');
sizeSlider.addEventListener('input', e => {
  pointSizeMultiplier = parseFloat(e.target.value);
  sizeVal.textContent = e.target.value;
});

init();
