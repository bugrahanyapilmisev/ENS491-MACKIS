// src/__tests__/mocks/server.ts
import { setupServer } from 'msw/node';
import { handlers } from './handlers';

/**
 * MSW Node.js sunucusu.
 * Vitest/jsdom ortamında fetch ve axios isteklerini intercept eder.
 * setup.ts'de beforeAll/afterEach/afterAll ile yönetilir.
 */
export const server = setupServer(...handlers);
