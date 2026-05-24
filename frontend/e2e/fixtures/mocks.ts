// e2e/fixtures/mocks.ts
// Tüm API mock'larını merkezi olarak yönetir.
// Her test dosyasından import edilir.

import { Page } from '@playwright/test';

// ─── Sabit Mock Veriler ───────────────────────────────────────────────────────

export const MOCK_TOKEN = 'e2e-mock-jwt-token';

export const loginSuccessResponse = {
  access_token: MOCK_TOKEN,
  token_type: 'bearer',
  user_name: 'Test Öğrencisi',
  is_admin: false,
};

export const adminLoginResponse = {
  access_token: MOCK_TOKEN,
  token_type: 'bearer',
  user_name: 'Test Admin',
  is_admin: true,
};

export const chatResponse = {
  answer: 'Erasmus değişim programı için minimum GNO 2.5 olmalıdır.',
  conversation_id: 42,
  query_id: 1,
  message_id: 1,
  confidence: 0.92,
  sources: [
    {
      chunk_id: 101,
      title: 'Erasmus Rehberi',
      excerpt: 'Erasmus başvurusu için minimum GNO şartı 2.5 olmalıdır.',
      url: 'https://mysu.sabanciuniv.edu/erasmus',
      score: 0.95,
    },
  ],
};

export const historyResponse = [
  {
    id: 1,
    title: 'Erasmus GNO sorusu',
    timestamp: new Date().toISOString(),
    preview: 'Erasmus için GNO kaç?',
    messages: [
      { id: 1, role: 'user',      content: 'Erasmus için GNO kaç?',        timestamp: new Date().toISOString(), sources: [], confidence: null },
      { id: 2, role: 'assistant', content: 'Minimum GNO 2.5 olmalıdır.',   timestamp: new Date().toISOString(), sources: [], confidence: 0.92 },
    ],
  },
  {
    id: 2,
    title: 'Kütüphane sorusu',
    timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(), // 2 gün önce
    preview: 'Kaç kitap ödünç alabilirim?',
    messages: [],
  },
];

export const statsResponse = { document_count: 42, topic_count: 17 };

// ─── Route Kurulum Fonksiyonları ─────────────────────────────────────────────

/** Başarılı normal kullanıcı girişi mock'u */
export async function mockLoginSuccess(page: Page) {
  await page.route('**/auth/login', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(loginSuccessResponse),
    });
  });
}

/** Admin kullanıcı girişi mock'u */
export async function mockAdminLogin(page: Page) {
  await page.route('**/auth/login', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(adminLoginResponse),
    });
  });
}

/** 401 Unauthorized girişi mock'u */
export async function mockLoginUnauthorized(page: Page) {
  await page.route('**/auth/login', async (route) => {
    await route.fulfill({
      status: 401,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'E-posta veya şifre hatalı' }),
    });
  });
}

/** Ağ hatası — sunucuya ulaşılamıyor */
export async function mockLoginNetworkError(page: Page) {
  await page.route('**/auth/login', async (route) => {
    await route.abort('failed');
  });
}

/** Chat endpoint mock'u */
export async function mockChat(page: Page, response = chatResponse) {
  await page.route('**/chat', async (route) => {
    if (route.request().method() === 'POST') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(response),
      });
    } else {
      await route.continue();
    }
  });
}

/** Chat history mock'u */
export async function mockHistory(page: Page, response = historyResponse) {
  await page.route('**/chat/history', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(response),
    });
  });
}

/** Stats mock'u */
export async function mockStats(page: Page) {
  await page.route('**/api/stats', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(statsResponse),
    });
  });
}

/** Tüm API'ları bir arada mock'la (chat akışları için) */
export async function mockAllApis(page: Page, opts: { isAdmin?: boolean } = {}) {
  if (opts.isAdmin) {
    await mockAdminLogin(page);
  } else {
    await mockLoginSuccess(page);
  }
  await mockHistory(page);
  await mockChat(page);
  await mockStats(page);
}

/** localStorage'a token set ederek login adımını atla */
export async function setAuthToken(page: Page) {
  await page.evaluate((token) => {
    localStorage.setItem('access_token', token);
  }, MOCK_TOKEN);
}
