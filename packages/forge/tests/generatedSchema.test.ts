import { RELEASE1_OP_SCHEMA } from '../src/generated/opSchema';

describe('Release 1 Op Schema Contract', () => {
  it('defines all required Release 1 operations', () => {
    const requiredOps = [
      'add', 'sub', 'mul', 'div', 'neg',
      'matmul', 'transpose', 'reshape', 'sum',
      'relu', 'relu_backward', 'mse_loss', 'mse_loss_backward', 'axpy'
    ];
    for (const op of requiredOps) {
      expect(RELEASE1_OP_SCHEMA).toHaveProperty(op);
      expect(RELEASE1_OP_SCHEMA[op].dtypes).toContain('float32');
    }
  });

  it('contains valid parameter definitions', () => {
    expect(RELEASE1_OP_SCHEMA['matmul'].params).toEqual([
      { name: 'M', type: 'positive-int' },
      { name: 'N', type: 'positive-int' },
      { name: 'K', type: 'positive-int' },
    ]);
    expect(RELEASE1_OP_SCHEMA['axpy'].params).toEqual([
      { name: 'numElements', type: 'positive-int' },
      { name: 'alpha', type: 'float32' },
    ]);
  });
});
