/**
 * AMEVA-Forge Lightweight In-Browser Visual Inspector & DevTools HUD
 * Real-time VRAM allocation tracking & Training loss curve visualization
 */
export interface InspectorState {
    mounted: boolean;
    history: Array<{
        step: number;
        loss: number;
    }>;
}
/**
 * Record a training step loss for live chart visualization
 */
export declare function recordStepLoss(step: number, loss: number): void;
/**
 * Clear recorded training history
 */
export declare function clearStepLossHistory(): void;
/**
 * Mount floating DevTools HUD overlay into DOM
 */
export declare function mountInspector(targetParent?: HTMLElement): HTMLElement;
/**
 * Unmount and destroy DevTools HUD
 */
export declare function unmountInspector(): void;
