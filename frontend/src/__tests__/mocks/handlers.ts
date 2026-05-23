// src/__tests__/mocks/handlers.ts
/**
 * MSW request handler'ları.
 *
 * İki farklı base URL kullanılıyor:
 *   - http://localhost/...      → KnowledgeBaseStats gibi fetch('/api/...') çağrıları
 *                                 (jsdom'da relative URL → window.location = localhost)
 *   - http://localhost:8000/... → axios tabanlı çağrılar (api/axios.ts baseURL)
 */
import { http, HttpResponse } from 'msw';

// ── Başarılı senaryo cevapları ────────────────────────────────────────────────

const STATS_RESPONSE = { document_count: 42, topic_count: 17 };

const SUGGESTIONS_RESPONSE = {
  suggestions: [
    { question: 'Erasmus GNO kaç olmalı?',          hint: 'Değişim',  emoji: '✈️' },
    { question: 'Kütüphane iade süresi ne kadar?',  hint: 'Kütüphane', emoji: '📚' },
    { question: 'Uyarı cezası ne zaman verilir?',   hint: 'Disiplin',  emoji: '⚖️' },
    { question: 'Burs başvuru tarihi ne zaman?',    hint: 'Burs',      emoji: '🎓' },
    { question: 'Çift anadal nasıl başvurulur?',    hint: 'Akademik',  emoji: '📝' },
    { question: 'Mezuniyet için kaç kredi gerekir?', hint: 'Mezuniyet', emoji: '🏛️' },
  ],
};

const LOGIN_SUCCESS_RESPONSE = {
  access_token: 'mock-jwt-token-for-testing',
  token_type: 'bearer',
  user_name: 'Test Öğrencisi',
  is_admin: false,
};

const ADMIN_LOGIN_RESPONSE = {
  access_token: 'mock-admin-jwt-token',
  token_type: 'bearer',
  user_name: 'Test Admin',
  is_admin: true,
};

const CHAT_HISTORY_RESPONSE = [
  {
    id: 1,
    title: 'Erasmus Soruları',
    timestamp: new Date().toISOString(),
    preview: 'Erasmus başvurusu için...',
    messages: [
      { id: 1, role: 'user',      content: 'GNO kaç olmalı?',    timestamp: new Date().toISOString() },
      { id: 2, role: 'assistant', content: 'En az 2.5 olmalıdır.', timestamp: new Date().toISOString(), confidence: 0.95 },
    ],
  },
];

// ── Handler tanımları ─────────────────────────────────────────────────────────

export const handlers = [
  // --- fetch('/api/stats') — KnowledgeBaseStats tarafından kullanılır
  http.get('http://localhost/api/stats', () =>
    HttpResponse.json(STATS_RESPONSE)
  ),

  // --- fetch('/api/suggestions')
  http.get('http://localhost/api/suggestions', () =>
    HttpResponse.json(SUGGESTIONS_RESPONSE)
  ),

  // --- axios POST /auth/login — başarılı giriş (default)
  http.post('http://localhost:8000/auth/login', () =>
    HttpResponse.json(LOGIN_SUCCESS_RESPONSE)
  ),

  // --- axios GET /chat/history
  http.get('http://localhost:8000/chat/history', () =>
    HttpResponse.json(CHAT_HISTORY_RESPONSE)
  ),

  // --- axios POST /chat
  http.post('http://localhost:8000/chat', () =>
    HttpResponse.json({
      answer: 'Mock RAG cevabı.',
      sources: [],
      conversation_id: 1,
      query_id: 1,
      message_id: 2,
      confidence: 0.95,
    })
  ),
];

// ── Özel handler'lar (belirli testlerde server.use(...) ile override edilir) ──

/** 401 döndüren login handler — yanlış şifre testi için */
export const loginUnauthorizedHandler = http.post(
  'http://localhost:8000/auth/login',
  () => HttpResponse.json({ detail: 'E-posta veya şifre hatalı' }, { status: 401 })
);

/** Ağ hatası simülasyonu — bağlantı kurulamadı testi için */
export const loginNetworkErrorHandler = http.post(
  'http://localhost:8000/auth/login',
  () => HttpResponse.error()
);

/** /api/stats için hata cevabı — KnowledgeBaseStats hata durumu testi */
export const statsErrorHandler = http.get(
  'http://localhost/api/stats',
  () => HttpResponse.json({ detail: 'Internal Server Error' }, { status: 500 })
);

/** Admin girişi — is_admin: true */
export const adminLoginHandler = http.post(
  'http://localhost:8000/auth/login',
  () => HttpResponse.json(ADMIN_LOGIN_RESPONSE)
);
