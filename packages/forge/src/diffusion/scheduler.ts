/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-328 Stable Diffusion Latent Scheduler & Asynchronous Yielding Loop
 *
 * WHAT: 디퓨전 타임스텝 스케줄링(Euler/LCM) 및 브라우저 TDR(Timeout Detection & Recovery) 방어 비동기 스케줄러입니다.
 * WHY: 16단계 디노이징 과정에서 OS GPU 드라이버(Windows 2초 제한)가 브라우저 탭을 강제 종료하는 것을 막고,
 *      부드러운 실시간 프로그레스 업데이트와 가우시안 잠재 노이즈 생성을 보장하기 위해 존재합니다.
 * HOW: 선형 베타 스케줄(beta_start=0.00085, beta_end=0.012)을 기반으로 alpha, sigma를 산출하며,
 *      각 디노이징 단계마다 requestAnimationFrame / setTimeout(0)으로 메인 스레드에 제어권을 양보(Yielding)합니다.
 */

export interface SchedulerStepOutput {
  prevSample: Float32Array;
  predOriginalSample?: Float32Array;
}

export class EulerDiscreteScheduler {
  public numSteps: number;
  public timesteps: number[] = [];
  public sigmas: Float32Array = new Float32Array(0);
  private numTrainTimesteps: number = 1000;
  private betas: Float32Array;
  private alphas: Float32Array;
  private alphasCumprod: Float32Array;

  constructor(numSteps: number = 4, betaStart: number = 0.00085, betaEnd: number = 0.012) {
    this.numSteps = numSteps;
    this.betas = new Float32Array(this.numTrainTimesteps);
    this.alphas = new Float32Array(this.numTrainTimesteps);
    this.alphasCumprod = new Float32Array(this.numTrainTimesteps);

    // 1. Scaled Linear Beta Schedule (SD 1.5 / SD-Turbo 표준)
    const start = Math.sqrt(betaStart);
    const end = Math.sqrt(betaEnd);
    let cumprod = 1.0;
    for (let i = 0; i < this.numTrainTimesteps; i++) {
      const t = i / (this.numTrainTimesteps - 1);
      const beta = Math.pow(start + t * (end - start), 2);
      this.betas[i] = beta;
      this.alphas[i] = 1.0 - beta;
      cumprod *= this.alphas[i];
      this.alphasCumprod[i] = cumprod;
    }

    this.setTimesteps(numSteps);
  }

  /**
   * 타임스텝 시퀀스를 설정하고 각 스텝별 sigma 값을 사전 계산합니다.
   */
  public setTimesteps(numSteps: number): void {
    this.numSteps = numSteps;
    this.timesteps = [];
    const stepRatio = Math.floor(this.numTrainTimesteps / numSteps);

    for (let i = 0; i < numSteps; i++) {
      this.timesteps.push((numSteps - 1 - i) * stepRatio);
    }

    // sigmas: sqrt((1 - alpha_prod) / alpha_prod)
    this.sigmas = new Float32Array(numSteps + 1);
    for (let i = 0; i < numSteps; i++) {
      const t = this.timesteps[i];
      const alphaProd = this.alphasCumprod[t];
      this.sigmas[i] = Math.sqrt((1.0 - alphaProd) / alphaProd);
    }
    this.sigmas[numSteps] = 0.0;
  }

  /**
   * 단일 디노이징 스텝 연산: x_{t-1} = x_t + dt * derivative
   */
  public step(
    modelOutput: Float32Array,
    stepIndex: number,
    sample: Float32Array
  ): SchedulerStepOutput {
    const sigma = this.sigmas[stepIndex];
    const sigmaNext = this.sigmas[stepIndex + 1];
    const dt = sigmaNext - sigma;

    const len = sample.length;
    const prevSample = new Float32Array(len);

    // Euler step: prev = sample + dt * modelOutput
    for (let i = 0; i < len; i++) {
      prevSample[i] = sample[i] + dt * modelOutput[i];
    }

    return { prevSample };
  }

  /**
   * 결정론적 시드 기반 표준 정규분포(가우시안) 잠재 노이즈 생성 (Box-Muller 변환)
   */
  public generateInitialNoise(channels: number, height: number, width: number, seed: number = 42): Float32Array {
    const totalElements = channels * height * width;
    const noise = new Float32Array(totalElements);

    // LCG PRNG
    let s = seed;
    const lcg = () => {
      s = (s * 1664525 + 1013904223) % 4294967296;
      return s / 4294967296;
    };

    for (let i = 0; i < totalElements; i += 2) {
      const u1 = Math.max(1e-7, lcg());
      const u2 = lcg();
      const mag = Math.sqrt(-2.0 * Math.log(u1));
      const z0 = mag * Math.cos(2.0 * Math.PI * u2);
      const z1 = mag * Math.sin(2.0 * Math.PI * u2);

      noise[i] = z0;
      if (i + 1 < totalElements) {
        noise[i + 1] = z1;
      }
    }

    return noise;
  }

  /**
   * 브라우저 TDR 크래시 방지 및 UI 이벤트 루프 양보 (Asynchronous Yielding)
   */
  public async yieldToMainThread(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }
}
