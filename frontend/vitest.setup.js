import '@testing-library/jest-dom'

process.env.NEXT_PUBLIC_SUPABASE_URL = 'http://localhost:54321';
process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY = 'dummy-key';

// On Node 25+, the jsdom environment drops window.localStorage/sessionStorage
// (vitest-dev/vitest#10867). Restore a prototype-based Web Storage so key
// persistence and Storage.prototype spies keep working.
if (typeof globalThis.localStorage === 'undefined') {
    class StorageShim {
        constructor() { this._store = new Map(); }
        getItem(key) { return this._store.has(String(key)) ? this._store.get(String(key)) : null; }
        setItem(key, value) { this._store.set(String(key), String(value)); }
        removeItem(key) { this._store.delete(String(key)); }
        clear() { this._store.clear(); }
        key(index) { return Array.from(this._store.keys())[index] ?? null; }
        get length() { return this._store.size; }
    }
    Object.defineProperty(globalThis, 'Storage', { value: StorageShim, configurable: true, writable: true });
    Object.defineProperty(globalThis, 'localStorage', { value: new StorageShim(), configurable: true });
    Object.defineProperty(globalThis, 'sessionStorage', { value: new StorageShim(), configurable: true });
    if (globalThis.window) {
        Object.defineProperty(globalThis.window, 'Storage', { value: StorageShim, configurable: true, writable: true });
        Object.defineProperty(globalThis.window, 'localStorage', { value: globalThis.localStorage, configurable: true });
        Object.defineProperty(globalThis.window, 'sessionStorage', { value: globalThis.sessionStorage, configurable: true });
    }
}
