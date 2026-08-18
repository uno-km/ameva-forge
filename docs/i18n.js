/**
 * ==============================================================================
 * AMEVA-Forge Documentation - Enterprise Multilingual (i18n) Core Engine
 * ==============================================================================
 * 
 * WHAT:
 *   Object-Oriented Client-Side Internationalization (i18n) Engine with
 *   Multi-Tier Persistent Storage (LocalStorage -> IndexedDB -> Cookie -> SessionStorage).
 * 
 * WHY:
 *   Allows users to seamlessly switch documentation across 6 languages:
 *   English (en), Korean (ko), Chinese (zh), Japanese (ja), Hindi (hi), Spanish (es).
 *   Guarantees persistent state across page transitions, reloads, and browser sessions.
 * 
 * HOW:
 *   1. I18nStorageAdapter: Provides unified async/sync storage access with graceful fallbacks.
 *   2. ForgeI18n: Core singleton managing translation dictionaries, DOM binding,
 *      event propagation, and language selector UI components.
 *   3. Auto-initializes on DOMContentLoaded with instant flash-free rendering.
 * 
 * @license Apache-2.0
 */

(function(global) {
    'use strict';

    /**
     * Supported Language Definition
     */
    const SUPPORTED_LANGUAGES = {
        'en': { code: 'en', name: 'English', nativeName: 'English', flag: '🇺🇸' },
        'ko': { code: 'ko', name: 'Korean', nativeName: '한국어', flag: '🇰🇷' },
        'zh': { code: 'zh', name: 'Chinese', nativeName: '简体中文', flag: '🇨🇳' },
        'ja': { code: 'ja', name: 'Japanese', nativeName: '日本語', flag: '🇯🇵' },
        'hi': { code: 'hi', name: 'Hindi', nativeName: 'हिन्दी', flag: '🇮🇳' },
        'es': { code: 'es', name: 'Spanish', nativeName: 'Español', flag: '🇪🇸' }
    };

    const DEFAULT_LANG = 'en';
    const STORAGE_KEY = 'ameva_forge_doc_lang';
    const DB_NAME = 'AmevaForgeDocDB';
    const DB_STORE_NAME = 'UserSettings';

    /**
     * ==========================================================================
     * 1. Multi-Tier Persistent Storage Adapter
     * ==========================================================================
     */
    class I18nStorageAdapter {
        constructor() {
            this.dbPromise = this._initIndexedDB();
        }

        _initIndexedDB() {
            return new Promise((resolve) => {
                if (!('indexedDB' in global)) {
                    resolve(null);
                    return;
                }
                try {
                    const req = global.indexedDB.open(DB_NAME, 1);
                    req.onupgradeneeded = (e) => {
                        const db = e.target.result;
                        if (!db.objectStoreNames.contains(DB_STORE_NAME)) {
                            db.createObjectStore(DB_STORE_NAME, { keyPath: 'key' });
                        }
                    };
                    req.onsuccess = (e) => resolve(e.target.result);
                    req.onerror = () => resolve(null);
                } catch (e) {
                    resolve(null);
                }
            });
        }

        /**
         * Smart Geo & Locale Auto-Detector for First-Time Visitors
         */
        detectLocaleFromEnvironment(customEnv = {}) {
            // 1. Inspect navigator.languages priority list
            try {
                const nav = customEnv.navigator || (typeof navigator !== 'undefined' ? navigator : {});
                const navLanguages = nav.languages || [nav.language || nav.userLanguage || ''];
                for (const fullLang of navLanguages) {
                    if (!fullLang) continue;
                    const lang = fullLang.toLowerCase();
                    if (lang.startsWith('ko')) return 'ko';
                    if (lang.startsWith('ja')) return 'ja';
                    if (lang.startsWith('zh')) return 'zh';
                    if (lang.startsWith('hi')) return 'hi';
                    if (lang.startsWith('es')) return 'es';
                    if (lang.startsWith('en')) return 'en';
                }
            } catch (e) {}

            // 2. Inspect Client Timezone
            try {
                const intlObj = customEnv.Intl || (typeof Intl !== 'undefined' ? Intl : null);
                const tz = intlObj?.DateTimeFormat?.()?.resolvedOptions?.()?.timeZone;
                if (tz) {
                    if (tz === 'Asia/Seoul' || tz === 'Asia/Pyongyang') return 'ko';
                    if (tz === 'Asia/Tokyo') return 'ja';
                    if (tz.includes('Shanghai') || tz.includes('Beijing') || tz.includes('Chongqing') || tz.includes('Hong_Kong') || tz.includes('Taipei')) return 'zh';
                    if (tz.includes('Kolkata') || tz.includes('Calcutta')) return 'en'; // Developer/Tech standard in India
                    if (tz.includes('Madrid') || tz.includes('Mexico') || tz.includes('Bogota') || tz.includes('Buenos_Aires') || tz.includes('Santiago') || tz.includes('Lima') || tz.includes('Caracas')) return 'es';
                }
            } catch (e) {}

            return DEFAULT_LANG;
        }

        /**
         * Asynchronous Country Code IP Detection (Optional fallback for first visit)
         */
        async detectCountryFromIpAsync() {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 1200); // 1.2s fast timeout
                const res = await fetch('https://api.country.is/', { signal: controller.signal });
                clearTimeout(timeoutId);
                if (res.ok) {
                    const data = await res.json();
                    const country = (data.country || '').toUpperCase();
                    const COUNTRY_MAP = {
                        'KR': 'ko',
                        'JP': 'ja',
                        'CN': 'zh', 'TW': 'zh', 'HK': 'zh', 'MO': 'zh',
                        'IN': 'en', // Developer/Tech standard in India
                        'ES': 'es', 'MX': 'es', 'AR': 'es', 'CO': 'es', 'CL': 'es', 'PE': 'es',
                        'VE': 'es', 'EC': 'es', 'GT': 'es', 'CU': 'es', 'BO': 'es', 'DO': 'es',
                        'HN': 'es', 'PY': 'es', 'SV': 'es', 'NI': 'es', 'CR': 'es', 'PR': 'es', 'PA': 'es', 'UY': 'es'
                    };
                    if (COUNTRY_MAP[country]) {
                        return COUNTRY_MAP[country];
                    }
                }
            } catch (e) {}
            return null;
        }

        getSync() {
            // 1. Check LocalStorage (Explicit User Choice)
            try {
                const val = localStorage.getItem(STORAGE_KEY);
                if (val && SUPPORTED_LANGUAGES[val]) return val;
            } catch (e) {}

            // 2. Check Cookie (Explicit User Choice)
            try {
                const match = document.cookie.match(new RegExp('(^| )' + STORAGE_KEY + '=([^;]+)'));
                if (match && SUPPORTED_LANGUAGES[match[2]]) return match[2];
            } catch (e) {}

            // 3. Check SessionStorage
            try {
                const val = sessionStorage.getItem(STORAGE_KEY);
                if (val && SUPPORTED_LANGUAGES[val]) return val;
            } catch (e) {}

            // 4. First-time visit: Auto-detect based on browser languages and timezone
            const detected = this.detectLocaleFromEnvironment();
            // Automatically persist detected first-time language
            this.set(detected);
            return detected;
        }

        async getAsync() {
            // 1. If explicit choice exists in storage, return it immediately
            try {
                const val = localStorage.getItem(STORAGE_KEY);
                if (val && SUPPORTED_LANGUAGES[val]) return val;
            } catch (e) {}

            // 2. Try reading from IndexedDB
            try {
                const db = await this.dbPromise;
                if (db) {
                    const idbVal = await new Promise((resolve) => {
                        const tx = db.transaction([DB_STORE_NAME], 'readonly');
                        const store = tx.objectStore(DB_STORE_NAME);
                        const req = store.get(STORAGE_KEY);
                        req.onsuccess = () => resolve(req.result?.value || null);
                        req.onerror = () => resolve(null);
                    });
                    if (idbVal && SUPPORTED_LANGUAGES[idbVal]) return idbVal;
                }
            } catch (e) {}

            // 3. If first visit, attempt IP country detection
            const ipLang = await this.detectCountryFromIpAsync();
            if (ipLang && SUPPORTED_LANGUAGES[ipLang]) {
                this.set(ipLang);
                return ipLang;
            }

            return this.getSync();
        }

        set(lang) {
            if (!SUPPORTED_LANGUAGES[lang]) return;

            // 1. Save to LocalStorage
            try {
                localStorage.setItem(STORAGE_KEY, lang);
            } catch (e) {}

            // 2. Save to Cookie (1 year duration, path=/)
            try {
                const date = new Date();
                date.setTime(date.getTime() + (365 * 24 * 60 * 60 * 1000));
                document.cookie = STORAGE_KEY + '=' + lang + '; expires=' + date.toUTCString() + '; path=/; SameSite=Lax';
            } catch (e) {}

            // 3. Save to SessionStorage
            try {
                sessionStorage.setItem(STORAGE_KEY, lang);
            } catch (e) {}

            // 4. Save to IndexedDB asynchronously
            this.dbPromise.then((db) => {
                if (db) {
                    try {
                        const tx = db.transaction([DB_STORE_NAME], 'readwrite');
                        const store = tx.objectStore(DB_STORE_NAME);
                        store.put({ key: STORAGE_KEY, value: lang, updatedAt: Date.now() });
                    } catch (e) {}
                }
            }).catch(() => {});
        }
    }

    /**
     * ==========================================================================
     * 2. Core OOP i18n Engine
     * ==========================================================================
     */
    class ForgeI18n {
        constructor() {
            this.storage = new I18nStorageAdapter();
            this.translations = {};
            this.currentLang = this.storage.getSync();
            this.listeners = [];
            this.isInitialized = false;
        }

        /**
         * Register translation dictionary for specific language
         * @param {string} lang Language code (e.g. 'en', 'ko')
         * @param {Object} dict Dictionary object
         */
        registerTranslations(lang, dict) {
            if (!this.translations[lang]) {
                this.translations[lang] = {};
            }
            this._deepMerge(this.translations[lang], dict);
        }

        _deepMerge(target, source) {
            for (const key of Object.keys(source)) {
                if (source[key] instanceof Object && !Array.isArray(source[key])) {
                    if (!target[key]) Object.assign(target, { [key]: {} });
                    this._deepMerge(target[key], source[key]);
                } else {
                    Object.assign(target, { [key]: source[key] });
                }
            }
        }

        /**
         * Initialize i18n engine and bind DOM
         */
        async init() {
            // Merge pending translations if loaded before core
            if (global.__FORGE_I18N_PENDING_TRANSLATIONS__) {
                for (const [lang, dict] of Object.entries(global.__FORGE_I18N_PENDING_TRANSLATIONS__)) {
                    this.registerTranslations(lang, dict);
                }
            }

            // Check async storage in background in case LocalStorage was empty
            const asyncLang = await this.storage.getAsync();
            if (asyncLang && asyncLang !== this.currentLang && SUPPORTED_LANGUAGES[asyncLang]) {
                this.currentLang = asyncLang;
            }

            this.applyToDOM();
            this.renderHeaderSelector();
            this.isInitialized = true;

            // Set document lang attribute
            document.documentElement.lang = this.currentLang;
        }

        /**
         * Retrieve translated text by dot-notated key
         * @param {string} key e.g. 'nav.home' or 'index.hero.title'
         * @param {Object} [params] Parameters for placeholder substitution
         * @returns {string}
         */
        t(key, params = {}) {
            let val = this._lookupKey(this.currentLang, key);
            if (val === undefined && this.currentLang !== DEFAULT_LANG) {
                // Fallback to default language (en)
                val = this._lookupKey(DEFAULT_LANG, key);
            }
            if (val === undefined) {
                return key; // return key as last resort
            }

            if (typeof val === 'string' && Object.keys(params).length > 0) {
                return this._interpolate(val, params);
            }
            return val;
        }

        _lookupKey(lang, keyPath) {
            const dict = this.translations[lang];
            if (!dict) return undefined;

            const parts = keyPath.split('.');
            let curr = dict;
            for (const part of parts) {
                if (curr === undefined || curr === null) return undefined;
                curr = curr[part];
            }
            return curr;
        }

        _interpolate(template, params) {
            return template.replace(/\{(\w+)\}/g, (match, p1) => {
                return params[p1] !== undefined ? params[p1] : match;
            });
        }

        /**
         * Switch current language and update entire DOM
         * @param {string} lang Target language code
         */
        setLanguage(lang) {
            if (!SUPPORTED_LANGUAGES[lang]) {
                console.warn(`[ForgeI18n] Unsupported language requested: ${lang}`);
                return;
            }

            this.currentLang = lang;
            this.storage.set(lang);
            document.documentElement.lang = lang;

            this.applyToDOM();
            this._notifyListeners(lang);
        }

        getLanguage() {
            return this.currentLang;
        }

        getSupportedLanguages() {
            return SUPPORTED_LANGUAGES;
        }

        /**
         * Subscribe to language change events
         */
        onLanguageChanged(callback) {
            if (typeof callback === 'function') {
                this.listeners.push(callback);
            }
        }

        _notifyListeners(lang) {
            for (const listener of this.listeners) {
                try {
                    listener(lang);
                } catch (e) {
                    console.error('[ForgeI18n] Listener error:', e);
                }
            }
            // Dispatch window custom event
            try {
                window.dispatchEvent(new CustomEvent('forge:languageChanged', { detail: { language: lang } }));
            } catch (e) {}
        }

        /**
         * Traverse DOM and update elements with data-i18n attributes
         * @param {HTMLElement|Document} [root=document]
         */
        applyToDOM(root = document) {
            // 1. Text content nodes [data-i18n]
            const textNodes = root.querySelectorAll('[data-i18n]');
            textNodes.forEach((el) => {
                const key = el.getAttribute('data-i18n');
                if (key) {
                    const translated = this.t(key);
                    if (translated !== undefined) {
                        el.textContent = translated;
                    }
                }
            });

            // 2. HTML content nodes [data-i18n-html] (for styled text, inline tags)
            const htmlNodes = root.querySelectorAll('[data-i18n-html]');
            htmlNodes.forEach((el) => {
                const key = el.getAttribute('data-i18n-html');
                if (key) {
                    const translated = this.t(key);
                    if (translated !== undefined) {
                        el.innerHTML = translated;
                    }
                }
            });

            // 3. Attribute nodes [data-i18n-attr="attrName:key,attrName2:key2"]
            const attrNodes = root.querySelectorAll('[data-i18n-attr]');
            attrNodes.forEach((el) => {
                const spec = el.getAttribute('data-i18n-attr');
                if (spec) {
                    const pairs = spec.split(';');
                    for (const pair of pairs) {
                        const [attr, key] = pair.split(':').map(s => s.trim());
                        if (attr && key) {
                            const translated = this.t(key);
                            if (translated !== undefined) {
                                el.setAttribute(attr, translated);
                            }
                        }
                    }
                }
            });

            // 4. Placeholders & Titles [data-i18n-placeholder], [data-i18n-title]
            root.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
                const key = el.getAttribute('data-i18n-placeholder');
                if (key) el.setAttribute('placeholder', this.t(key));
            });

            root.querySelectorAll('[data-i18n-title]').forEach(el => {
                const key = el.getAttribute('data-i18n-title');
                if (key) el.setAttribute('title', this.t(key));
            });

            // 5. Update <title> and <meta name="description"> if keys are specified
            const titleEl = root.querySelector('title[data-i18n]');
            if (titleEl) {
                const key = titleEl.getAttribute('data-i18n');
                if (key) document.title = this.t(key);
            }

            const metaDesc = root.querySelector('meta[name="description"][data-i18n]');
            if (metaDesc) {
                const key = metaDesc.getAttribute('data-i18n');
                if (key) metaDesc.setAttribute('content', this.t(key));
            }

            // 6. Update selector UI state if rendered
            const selector = root.querySelector('#forgeLanguageSelect');
            if (selector && selector.value !== this.currentLang) {
                selector.value = this.currentLang;
            }
        }

        /**
         * Render the Language Selector dropdown in the header
         */
        renderHeaderSelector() {
            const badgeGroup = document.querySelector('.header-badge-group');
            if (!badgeGroup) return;

            let container = document.getElementById('forgeI18nContainer');
            if (!container) {
                container = document.createElement('div');
                container.id = 'forgeI18nContainer';
                container.className = 'forge-i18n-selector-wrapper';
                // Insert at the front of header-badge-group
                badgeGroup.insertBefore(container, badgeGroup.firstChild);
            }

            const optionsHtml = Object.values(SUPPORTED_LANGUAGES).map(item => {
                const selected = item.code === this.currentLang ? 'selected' : '';
                return `<option value="${item.code}" ${selected}>${item.flag} ${item.nativeName}</option>`;
            }).join('');

            container.innerHTML = `
                <label for="forgeLanguageSelect" class="forge-i18n-label" title="Select Documentation Language">
                    <span class="forge-i18n-globe">🌐</span>
                </label>
                <select id="forgeLanguageSelect" class="forge-i18n-select" aria-label="Language Selector">
                    ${optionsHtml}
                </select>
            `;

            const selectEl = container.querySelector('#forgeLanguageSelect');
            if (selectEl) {
                selectEl.addEventListener('change', (e) => {
                    this.setLanguage(e.target.value);
                });
            }
        }
    }

    // Export Singleton Instance
    const i18n = new ForgeI18n();
    global.forgeI18n = i18n;

    // Auto-init on DOM ready or immediate if already ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => i18n.init());
    } else {
        i18n.init();
    }

})(typeof window !== 'undefined' ? window : this);
