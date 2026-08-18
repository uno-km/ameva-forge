/**
 * ==============================================================================
 * AMEVA-Forge i18n Runtime & Storage Tier Simulation Test
 * ==============================================================================
 */

const fs = require('fs');
const path = require('path');

const DOCS_DIR = path.resolve(__dirname, '../docs');
const TRANSLATIONS_PATH = path.join(DOCS_DIR, 'i18n-translations.js');
const I18N_CORE_PATH = path.join(DOCS_DIR, 'i18n.js');

// Mock Browser Environment
const mockLocalStorage = {
    _data: {},
    getItem(k) { return this._data[k] || null; },
    setItem(k, v) { this._data[k] = String(v); },
    removeItem(k) { delete this._data[k]; }
};

const mockDocument = {
    cookie: '',
    documentElement: { lang: 'en' },
    querySelectorAll(selector) { return []; },
    querySelector(selector) { return null; },
    createElement(tag) {
        return {
            id: '',
            className: '',
            innerHTML: '',
            querySelector() { return null; },
            addEventListener() {}
        };
    },
    addEventListener() {},
    readyState: 'complete'
};

const mockWindow = {
    localStorage: mockLocalStorage,
    document: mockDocument,
    navigator: { language: 'ko-KR' },
    addEventListener() {},
    dispatchEvent() {}
};

function runRuntimeTest() {
    console.log("================================================================================");
    console.log(" [AMEVA-Forge] Runtime Storage & Language Switch Simulation");
    console.log("================================================================================");

    const i18nCode = fs.readFileSync(I18N_CORE_PATH, 'utf8');
    const transCode = fs.readFileSync(TRANSLATIONS_PATH, 'utf8');

    const fnCore = new Function('window', 'global', 'localStorage', 'document', 'navigator', i18nCode);
    const fnTrans = new Function('window', 'global', 'localStorage', 'document', 'navigator', transCode);

    fnCore(mockWindow, mockWindow, mockLocalStorage, mockDocument, mockWindow.navigator);
    fnTrans(mockWindow, mockWindow, mockLocalStorage, mockDocument, mockWindow.navigator);

    const i18n = mockWindow.forgeI18n;
    if (!i18n) throw new Error("forgeI18n singleton not initialized");

    console.log("[*] Initial detected language (Navigator ko-KR):", i18n.getLanguage());
    if (i18n.getLanguage() !== 'ko') throw new Error(`Expected 'ko', got ${i18n.getLanguage()}`);
    console.log("[PASS] Navigator language fallback correctly detected 'ko'");

    // Test translation lookup in Korean
    const koTitle = i18n.t('home.welcomeTitle');
    console.log("[*] Korean Translation of 'home.welcomeTitle':", koTitle);
    if (!koTitle.includes('환영합니다')) throw new Error("Incorrect Korean translation");
    console.log("[PASS] Korean translation lookup validated");

    // Test language switch to Japanese
    console.log("[*] Switching language to 'ja' (Japanese)...");
    i18n.setLanguage('ja');
    if (mockLocalStorage.getItem('ameva_forge_doc_lang') !== 'ja') throw new Error("LocalStorage not updated");
    if (!mockDocument.cookie.includes('ameva_forge_doc_lang=ja')) throw new Error("Cookie not updated");
    if (mockDocument.documentElement.lang !== 'ja') throw new Error("HTML lang attribute not updated");
    console.log("[PASS] LocalStorage, Cookie, and HTML lang correctly updated to 'ja'");

    const jaTitle = i18n.t('home.welcomeTitle');
    console.log("[*] Japanese Translation of 'home.welcomeTitle':", jaTitle);
    if (!jaTitle.includes('ようこそ')) throw new Error("Incorrect Japanese translation");
    console.log("[PASS] Japanese translation lookup validated");

    // Test language switch to Hindi
    console.log("[*] Switching language to 'hi' (Hindi)...");
    i18n.setLanguage('hi');
    const hiTitle = i18n.t('home.welcomeTitle');
    console.log("[*] Hindi Translation of 'home.welcomeTitle':", hiTitle);
    if (!hiTitle.includes('स्वागत')) throw new Error("Incorrect Hindi translation");
    console.log("[PASS] Hindi translation lookup validated");

    // Test language switch to Spanish
    console.log("[*] Switching language to 'es' (Spanish)...");
    i18n.setLanguage('es');
    const esTitle = i18n.t('home.welcomeTitle');
    console.log("[*] Spanish Translation of 'home.welcomeTitle':", esTitle);
    if (!esTitle.includes('Bienvenido')) throw new Error("Incorrect Spanish translation");
    console.log("[PASS] Spanish translation lookup validated");

    // Test language switch to Chinese
    console.log("[*] Switching language to 'zh' (Chinese)...");
    i18n.setLanguage('zh');
    const zhTitle = i18n.t('home.welcomeTitle');
    console.log("[*] Chinese Translation of 'home.welcomeTitle':", zhTitle);
    if (!zhTitle.includes('欢迎')) throw new Error("Incorrect Chinese translation");
    console.log("[PASS] Chinese translation lookup validated");

    // Test language switch to English
    console.log("[*] Switching language to 'en' (English)...");
    i18n.setLanguage('en');
    const enTitle = i18n.t('home.welcomeTitle');
    console.log("[*] English Translation of 'home.welcomeTitle':", enTitle);
    if (!enTitle.includes('Welcome')) throw new Error("Incorrect English translation");
    console.log("[PASS] English translation lookup validated");

    // Test parameter interpolation
    const interp = i18n.t('playgrounds.predictionResult', { digit: 7, conf: '99.8' });
    console.log("[*] Parameter Interpolation Output:", interp);
    if (!interp.includes('Digit 7') || !interp.includes('99.8%')) throw new Error("Interpolation failed");
    console.log("[PASS] Parameter interpolation '{digit}', '{conf}' validated");

    // Test Smart First-Visit Auto-Detection for Various Locales & Timezones
    console.log("\n[*] Testing First-Visit Locale & Timezone Auto-Detection Matrix...");
    
    // Case A: Tokyo Timezone -> 'ja'
    const detectedJa = i18n.storage.detectLocaleFromEnvironment({
        navigator: { languages: ['ja-JP', 'ja', 'en'] },
        Intl: { DateTimeFormat: () => ({ resolvedOptions: () => ({ timeZone: 'Asia/Tokyo' }) }) }
    });
    console.log("  - Tokyo Environment -> Auto-detected:", detectedJa);
    if (detectedJa !== 'ja') throw new Error("Failed to auto-detect Japanese");

    // Case B: China/Shanghai Timezone -> 'zh'
    const detectedZh = i18n.storage.detectLocaleFromEnvironment({
        navigator: { languages: ['zh-CN', 'zh', 'en'] },
        Intl: { DateTimeFormat: () => ({ resolvedOptions: () => ({ timeZone: 'Asia/Shanghai' }) }) }
    });
    console.log("  - China Environment -> Auto-detected:", detectedZh);
    if (detectedZh !== 'zh') throw new Error("Failed to auto-detect Chinese");

    // Case C: Spain/Madrid Timezone -> 'es'
    const detectedEs = i18n.storage.detectLocaleFromEnvironment({
        navigator: { languages: ['es-ES', 'es', 'en'] },
        Intl: { DateTimeFormat: () => ({ resolvedOptions: () => ({ timeZone: 'Europe/Madrid' }) }) }
    });
    console.log("  - Spain Environment -> Auto-detected:", detectedEs);
    if (detectedEs !== 'es') throw new Error("Failed to auto-detect Spanish");

    // Case D: India/Kolkata Timezone (with generic/en browser language) -> 'en' (Tech/Developer standard)
    const detectedHi = i18n.storage.detectLocaleFromEnvironment({
        navigator: { languages: ['en-IN', 'en-US', 'en'] },
        Intl: { DateTimeFormat: () => ({ resolvedOptions: () => ({ timeZone: 'Asia/Kolkata' }) }) }
    });
    console.log("  - India Environment -> Auto-detected:", detectedHi);
    if (detectedHi !== 'en') throw new Error("Failed to auto-detect English for India environment");

    // Case E: Explicit Hindi browser preference -> 'hi'
    const detectedExplicitHi = i18n.storage.detectLocaleFromEnvironment({
        navigator: { languages: ['hi-IN', 'hi', 'en'] },
        Intl: { DateTimeFormat: () => ({ resolvedOptions: () => ({ timeZone: 'Asia/Kolkata' }) }) }
    });
    console.log("  - Explicit Hindi Browser Setting -> Auto-detected:", detectedExplicitHi);
    if (detectedExplicitHi !== 'hi') throw new Error("Failed to respect explicit Hindi preference");

    console.log("[PASS] All First-Visit Locale & Timezone Auto-Detection matrix verified");

    console.log("\n================================================================================");
    console.log(" >>> RUNTIME ENGINE & STORAGE ADAPTER TESTS 100% SUCCESSFUL <<<");
    console.log("================================================================================");
}

runRuntimeTest();

