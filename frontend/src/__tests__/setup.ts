// src/__tests__/setup.ts
/**
 * Vitest global setup dosyası.
 * vite.config.ts → test.setupFiles ile her test dosyasından önce çalışır.
 */
import '@testing-library/jest-dom';
import { server } from './mocks/server';

// MSW sunucusunu tüm test session'ı boyunca ayağa kaldır
beforeAll(() => server.listen({ onUnhandledRequest: 'warn' }));

// Her testten sonra handler'ları sıfırla (override'lar bir sonraki teste taşınmasın)
afterEach(() => server.resetHandlers());

// Tüm testler bitince sunucuyu kapat
afterAll(() => server.close());
