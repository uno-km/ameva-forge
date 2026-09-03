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
export declare class EulerDiscreteScheduler {
    numSteps: number;
    timesteps: number[];
    sigmas: Float32Array;
    private numTrainTimesteps;
    private betas;
    private alphas;
    private alphasCumprod;
    constructor(numSteps?: number, betaStart?: number, betaEnd?: number);
    /**
     * 타임스텝 시퀀스를 설정하고 각 스텝별 sigma 값을 사전 계산합니다.
     */
    setTimesteps(numSteps: number): void;
    /**
     * 단일 디노이징 스텝 연산: x_{t-1} = x_t + dt * derivative
     */
    step(modelOutput: Float32Array, stepIndex: number, sample: Float32Array): SchedulerStepOutput;
    /**
     * 결정론적 시드 기반 표준 정규분포(가우시안) 잠재 노이즈 생성 (Box-Muller 변환)
     */
    generateInitialNoise(channels: number, height: number, width: number, seed?: number): Float32Array;
    /**
     * 브라우저 TDR 크래시 방지 및 UI 이벤트 루프 양보 (Asynchronous Yielding)
     */
    yieldToMainThread(): Promise<void>;
}
