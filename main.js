// WebGPU 3D Gaussian Splatting Viewer
const canvas = document.getElementById('canvas');
const status = document.getElementById('status');
const fpsEl = document.getElementById('fps');
const loadBtn = document.getElementById('loadBtn');
const fileInput = document.getElementById('fileInput');

let device, context, format;
let camera = { x: 0, y: 0, z: 5, rotX: 0, rotY: 0 };
let isDragging = false;
let lastMouse = { x: 0, y: 0 };

// Gaussian data
let gaussians = null;
let vertexBuffer = null;
let pipeline = null;

// Init WebGPU
async function initWebGPU() {
  if (!navigator.gpu) {
    status.textContent = 'WebGPU not supported';
    status.style.color = '#f00';
    return false;
  }
  
  const adapter = await navigator.gpu.requestAdapter();
  device = await adapter.requestDevice();
  
  context = canvas.getContext('webgpu');
  format = navigator.gpu.getPreferredCanvasFormat();
  
  context.configure({ device, format, alphaMode: 'opaque' });
  
  resize();
  status.textContent = 'WebGPU ready. Load a .ply file';
  return true;
}

// Resize handler
function resize() {
  canvas.width = window.innerWidth * devicePixelRatio;
  canvas.height = window.innerHeight * devicePixelRatio;
}

// Shader code
const shaderCode = `
struct Uniforms {
  mvp: mat4x4f,
  viewport: vec2f,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
}

@vertex
fn vs_main(
  @location(0) pos: vec3f,
  @location(1) col: vec4f
) -> VertexOutput {
  var out: VertexOutput;
  out.position = uniforms.mvp * vec4f(pos, 1.0);
  out.color = col;
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  return in.color;
}
`;

// Parse PLY file
async function parsePLY(buffer) {
  const decoder = new TextDecoder();
  const text = decoder.decode(buffer);
  const lines = text.split('\n');
  
  let vertexCount = 0;
  let headerEnd = 0;
  
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('element vertex')) {
      vertexCount = parseInt(lines[i].split(' ')[2]);
    }
    if (lines[i] === 'end_header') {
      headerEnd = i + 1;
      break;
    }
  }
  
  status.textContent = `Parsing ${vertexCount} gaussians...`;
  return { vertexCount, headerEnd, lines };
}

// Create render pipeline
let uniformBuffer, bindGroup;

function createPipeline() {
  const shaderModule = device.createShaderModule({ code: shaderCode });
  
  uniformBuffer = device.createBuffer({
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: 'uniform' }
    }]
  });
  
  bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
  });
  
  pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 28,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x4' },
        ]
      }]
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [{ format }]
    },
    primitive: { topology: 'point-list' }
  });
}

// Matrix helpers
function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  return new Float32Array([
    f/aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far+near)/(near-far), -1,
    0, 0, 2*far*near/(near-far), 0
  ]);
}

function lookAt() {
  const cx = Math.cos(camera.rotX), sx = Math.sin(camera.rotX);
  const cy = Math.cos(camera.rotY), sy = Math.sin(camera.rotY);
  return new Float32Array([
    cy, sx*sy, -cx*sy, 0,
    0, cx, sx, 0,
    sy, -sx*cy, cx*cy, 0,
    0, 0, -camera.z, 1
  ]);
}

function mulMat4(a, b) {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      out[i*4+j] = a[i*4]*b[j] + a[i*4+1]*b[4+j] + a[i*4+2]*b[8+j] + a[i*4+3]*b[12+j];
    }
  }
  return out;
}

// FPS counter
let frameCount = 0, lastTime = performance.now();
function updateFPS() {
  frameCount++;
  const now = performance.now();
  if (now - lastTime >= 1000) {
    fpsEl.textContent = `${frameCount} FPS`;
    frameCount = 0;
    lastTime = now;
  }
}

// Render
let vertexCount = 0;
function render() {
  if (!pipeline || !vertexBuffer) {
    requestAnimationFrame(render);
    updateFPS();
    return;
  }
  
  const aspect = canvas.width / canvas.height;
  const proj = perspective(Math.PI / 4, aspect, 0.1, 1000);
  const view = lookAt();
  const mvp = mulMat4(proj, view);
  
  device.queue.writeBuffer(uniformBuffer, 0, mvp);
  
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.04, g: 0.04, b: 0.04, a: 1 },
      loadOp: 'clear',
      storeOp: 'store'
    }]
  });
  
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.draw(vertexCount);
  pass.end();
  
  device.queue.submit([encoder.finish()]);
  updateFPS();
  requestAnimationFrame(render);
}

// Mouse controls
canvas.addEventListener('mousedown', e => {
  isDragging = true;
  lastMouse = { x: e.clientX, y: e.clientY };
});

canvas.addEventListener('mousemove', e => {
  if (!isDragging) return;
  camera.rotY += (e.clientX - lastMouse.x) * 0.01;
  camera.rotX += (e.clientY - lastMouse.y) * 0.01;
  lastMouse = { x: e.clientX, y: e.clientY };
});

canvas.addEventListener('mouseup', () => isDragging = false);
canvas.addEventListener('mouseleave', () => isDragging = false);

canvas.addEventListener('wheel', e => {
  camera.z += e.deltaY * 0.01;
  camera.z = Math.max(1, Math.min(50, camera.z));
});

// File loading
loadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  
  status.textContent = 'Loading file...';
  const buffer = await file.arrayBuffer();
  await loadPLY(buffer);
});

async function loadPLY(buffer) {
  const { vertexCount: count, headerEnd, lines } = await parsePLY(buffer);
  vertexCount = count;
  
  // Parse vertices (position + color)
  const data = new Float32Array(vertexCount * 7);
  for (let i = 0; i < vertexCount; i++) {
    const parts = lines[headerEnd + i].trim().split(/\s+/);
    // x, y, z
    data[i*7] = parseFloat(parts[0]);
    data[i*7+1] = parseFloat(parts[1]);
    data[i*7+2] = parseFloat(parts[2]);
    // r, g, b, a (normalized)
    data[i*7+3] = (parseFloat(parts[3]) || 128) / 255;
    data[i*7+4] = (parseFloat(parts[4]) || 128) / 255;
    data[i*7+5] = (parseFloat(parts[5]) || 128) / 255;
    data[i*7+6] = 1.0;
  }
  
  vertexBuffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, data);
  
  status.textContent = `Loaded ${vertexCount} gaussians`;
}

// Resize
window.addEventListener('resize', resize);

// Init
async function init() {
  if (await initWebGPU()) {
    createPipeline();
    render();
  }
}

init();
