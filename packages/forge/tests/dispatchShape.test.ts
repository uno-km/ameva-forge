import { computeDispatch2D } from '../src/tensor/dispatchShape';
import { AMEVAForgeValidationError } from '../src/errors';

describe('2D Workgroup Dispatch Calculator', () => {
  it('handles small sizes <= 65535 workgroups as 1D dispatch', () => {
    // 64 elements -> 1 workgroup
    const d1 = computeDispatch2D(64, 64);
    expect(d1.dispatchX).toBe(1);
    expect(d1.dispatchY).toBe(1);
    expect(d1.workgroupsX).toBe(1);

    // 4,194,240 elements (65535 * 64) -> exact 65535 workgroups
    const d2 = computeDispatch2D(4_194_240, 64);
    expect(d2.dispatchX).toBe(65535);
    expect(d2.dispatchY).toBe(1);
    expect(d2.workgroupsX).toBe(65535);
  });

  it('splits large sizes > 4.19M elements into 2D dispatch without truncation', () => {
    // 4,300,000 elements -> ceil(4300000 / 64) = 67188 workgroups
    // dispatchX = 65535, dispatchY = ceil(67188 / 65535) = 2
    const d = computeDispatch2D(4_300_000, 64);
    expect(d.totalWorkgroups).toBe(67188);
    expect(d.dispatchX).toBe(65535);
    expect(d.dispatchY).toBe(2);
    expect(d.workgroupsX).toBe(65535);
    expect(d.dispatchX * d.dispatchY * 64).toBeGreaterThanOrEqual(4_300_000);
  });

  it('supports extreme sizes up to 100M elements', () => {
    // 100,000,000 elements -> 1,562,500 workgroups -> dispatchX=65535, dispatchY=24
    const d = computeDispatch2D(100_000_000, 64);
    expect(d.dispatchX).toBe(65535);
    expect(d.dispatchY).toBe(24);
    expect(d.dispatchX * d.dispatchY * 64).toBeGreaterThanOrEqual(100_000_000);
  });

  it('rejects invalid or negative numElements', () => {
    expect(() => computeDispatch2D(0, 64)).toThrow(AMEVAForgeValidationError);
    expect(() => computeDispatch2D(-100, 64)).toThrow(AMEVAForgeValidationError);
    expect(() => computeDispatch2D(1.5, 64)).toThrow(AMEVAForgeValidationError);
  });
});
