/**
 * AMEVA-Forge Lightweight In-Browser Visual Inspector & DevTools HUD
 * Real-time VRAM allocation tracking & Training loss curve visualization
 */

import { _globalQuotaManager, getQuotaSnapshot } from '../webgpu/quota';
import { _globalRegistry } from '../tensor/tensorRegistry';

export interface InspectorState {
  mounted: boolean;
  history: Array<{ step: number; loss: number }>;
}

let inspectorContainer: HTMLElement | null = null;
let canvasElement: HTMLCanvasElement | null = null;
let animationFrameId: number | null = null;
const lossHistory: Array<{ step: number; loss: number }> = [];

/**
 * Record a training step loss for live chart visualization
 */
export function recordStepLoss(step: number, loss: number): void {
  lossHistory.push({ step, loss });
  if (lossHistory.length > 200) {
    lossHistory.shift();
  }
}

/**
 * Clear recorded training history
 */
export function clearStepLossHistory(): void {
  lossHistory.length = 0;
}

/**
 * Render HUD loop
 */
function renderHUD(): void {
  if (!canvasElement) return;
  const ctx = canvasElement.getContext('2d');
  if (!ctx) return;

  const width = canvasElement.width;
  const height = canvasElement.height;

  // Background
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, width, height);

  // Header / Metrics
  const quota = getQuotaSnapshot();
  const handles = _globalRegistry.snapshotHandles();

  ctx.fillStyle = '#f8fafc';
  ctx.font = 'bold 12px monospace';
  ctx.fillText('⚡ AMEVA-Forge DevTools', 10, 20);

  ctx.font = '10px monospace';
  ctx.fillStyle = '#94a3b8';
  const vramKB = (quota.usedBytes / 1024).toFixed(1);
  const maxKB = (quota.maxBytes / (1024 * 1024)).toFixed(0);
  ctx.fillText(`VRAM: ${vramKB} KB / ${maxKB} MB | Handles: ${handles.length}`, 10, 36);

  // VRAM Bar
  const barWidth = width - 20;
  const barHeight = 6;
  ctx.fillStyle = '#334155';
  ctx.fillRect(10, 44, barWidth, barHeight);

  const usageRatio = Math.min(1.0, quota.usedBytes / Math.max(1, quota.maxBytes));
  ctx.fillStyle = usageRatio > 0.8 ? '#ef4444' : '#10b981';
  ctx.fillRect(10, 44, barWidth * usageRatio, barHeight);

  // Loss Curve Area
  const chartX = 10;
  const chartY = 60;
  const chartW = width - 20;
  const chartH = height - 70;

  ctx.fillStyle = '#1e293b';
  ctx.fillRect(chartX, chartY, chartW, chartH);

  if (lossHistory.length > 1) {
    const minLoss = Math.min(...lossHistory.map(h => h.loss));
    const maxLoss = Math.max(...lossHistory.map(h => h.loss), minLoss + 1e-4);

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    for (let i = 0; i < lossHistory.length; i++) {
      const x = chartX + (i / (lossHistory.length - 1)) * chartW;
      const normalizedY = (lossHistory[i].loss - minLoss) / (maxLoss - minLoss);
      const y = chartY + chartH - normalizedY * (chartH - 8) - 4;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    const latest = lossHistory[lossHistory.length - 1];
    ctx.fillStyle = '#38bdf8';
    ctx.font = '9px monospace';
    ctx.fillText(`Step ${latest.step}: Loss ${latest.loss.toFixed(4)}`, 14, chartY + 12);
  } else {
    ctx.fillStyle = '#64748b';
    ctx.font = '9px monospace';
    ctx.fillText('Awaiting training steps...', chartX + 10, chartY + chartH / 2);
  }

  animationFrameId = requestAnimationFrame(renderHUD);
}

/**
 * Mount floating DevTools HUD overlay into DOM
 */
export function mountInspector(targetParent?: HTMLElement): HTMLElement {
  if (inspectorContainer) {
    return inspectorContainer;
  }

  const container = document.createElement('div');
  container.id = 'ameva-forge-devtools';
  container.style.position = 'fixed';
  container.style.bottom = '16px';
  container.style.right = '16px';
  container.style.width = '280px';
  container.style.height = '180px';
  container.style.backgroundColor = '#0f172a';
  container.style.border = '1px solid #334155';
  container.style.borderRadius = '8px';
  container.style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.4)';
  container.style.zIndex = '999999';
  container.style.overflow = 'hidden';
  container.style.fontFamily = 'monospace';

  const canvas = document.createElement('canvas');
  canvas.width = 280;
  canvas.height = 180;
  canvas.style.display = 'block';
  container.appendChild(canvas);

  const parent = targetParent || document.body;
  parent.appendChild(container);

  inspectorContainer = container;
  canvasElement = canvas;

  if (typeof requestAnimationFrame !== 'undefined') {
    animationFrameId = requestAnimationFrame(renderHUD);
  }

  return container;
}

/**
 * Unmount and destroy DevTools HUD
 */
export function unmountInspector(): void {
  if (animationFrameId !== null && typeof cancelAnimationFrame !== 'undefined') {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  if (inspectorContainer && inspectorContainer.parentNode) {
    inspectorContainer.parentNode.removeChild(inspectorContainer);
  }
  inspectorContainer = null;
  canvasElement = null;
}
