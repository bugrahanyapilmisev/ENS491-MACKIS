// src/__tests__/api.test.ts
/**
 * lib/api.ts fonksiyonları için birim testler.
 * MSW axios isteklerini intercept eder — gerçek ağ isteği gitmez.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { sendMessageToRAG, fetchChatHistory, loginUser } from '../lib/api';
import { server } from './mocks/server';
import { loginUnauthorizedHandler } from './mocks/handlers';
import { http, HttpResponse } from 'msw';

beforeEach(() => {
  localStorage.clear();
});

// =============================================================================
// loginUser
// =============================================================================

describe('loginUser', () => {
  it('başarılı girişte access_token döner', async () => {
    const result = await loginUser('student@sabanciuniv.edu', 'password123');
    expect(result.access_token).toBe('mock-jwt-token-for-testing');
  });

  it('başarılı girişte user_name döner', async () => {
    const result = await loginUser('student@sabanciuniv.edu', 'password123');
    expect(result.user_name).toBe('Test Öğrencisi');
  });

  it('başarılı girişte token_type "bearer" döner', async () => {
    const result = await loginUser('student@sabanciuniv.edu', 'password123');
    expect(result.token_type).toBe('bearer');
  });

  it('başarılı girişte is_admin false döner', async () => {
    const result = await loginUser('student@sabanciuniv.edu', 'password123');
    expect(result.is_admin).toBe(false);
  });

  it('401 gelince AxiosError fırlatır', async () => {
    server.use(loginUnauthorizedHandler);
    await expect(loginUser('a@b.com', 'yanlis')).rejects.toThrow();
  });

  it('401 hatası 401 status koduna sahip', async () => {
    server.use(loginUnauthorizedHandler);
    try {
      await loginUser('a@b.com', 'yanlis');
    } catch (err: any) {
      expect(err.response?.status).toBe(401);
    }
  });
});

// =============================================================================
// sendMessageToRAG
// =============================================================================

describe('sendMessageToRAG', () => {
  beforeEach(() => {
    // Her testten önce token set et (axios interceptor JWT ekler)
    localStorage.setItem('access_token', 'mock-jwt-token-for-testing');
  });

  it('answer alanı döner', async () => {
    const result = await sendMessageToRAG('GNO kaç olmalı?');
    expect(result.answer).toBe('Mock RAG cevabı.');
  });

  it('conversation_id döner', async () => {
    const result = await sendMessageToRAG('test sorusu');
    expect(result.conversation_id).toBe(1);
  });

  it('confidence döner', async () => {
    const result = await sendMessageToRAG('test sorusu');
    expect(result.confidence).toBe(0.95);
  });

  it('sources array döner', async () => {
    const result = await sendMessageToRAG('test sorusu');
    expect(Array.isArray(result.sources)).toBe(true);
  });

  it('conversation_id payload\'a eklenir', async () => {
    // Handler'ı payload kontrolü yapacak şekilde override et
    let receivedBody: any = null;
    server.use(
      http.post('http://localhost:8000/chat', async ({ request }) => {
        receivedBody = await request.json();
        return HttpResponse.json({
          answer: 'test',
          sources: [],
          conversation_id: 5,
          query_id: 1,
          message_id: 1,
          confidence: 0.9,
        });
      })
    );

    await sendMessageToRAG('soru', 'qwen3-32b', 5);
    expect(receivedBody?.conversation_id).toBe(5);
  });

  it('conversation_id verilmezse payload\'a eklenmez', async () => {
    let receivedBody: any = null;
    server.use(
      http.post('http://localhost:8000/chat', async ({ request }) => {
        receivedBody = await request.json();
        return HttpResponse.json({
          answer: 'test',
          sources: [],
          conversation_id: 1,
          query_id: 1,
          message_id: 1,
          confidence: 0.9,
        });
      })
    );

    await sendMessageToRAG('soru');
    expect(receivedBody?.conversation_id).toBeUndefined();
  });
});

// =============================================================================
// fetchChatHistory
// =============================================================================

describe('fetchChatHistory', () => {
  beforeEach(() => {
    localStorage.setItem('access_token', 'mock-jwt-token-for-testing');
  });

  it('array döner', async () => {
    const result = await fetchChatHistory();
    expect(Array.isArray(result)).toBe(true);
  });

  it('sohbet verisi içerir', async () => {
    const result = await fetchChatHistory();
    expect(result.length).toBeGreaterThan(0);
    expect(result[0]).toHaveProperty('id');
    expect(result[0]).toHaveProperty('title');
    expect(result[0]).toHaveProperty('messages');
  });

  it('sohbet mesajları role alanı içerir', async () => {
    const result = await fetchChatHistory();
    const messages = result[0].messages ?? [];
    expect(messages.length).toBeGreaterThan(0);
    expect(messages[0]).toHaveProperty('role');
  });

  it('boş history durumunda boş array döner', async () => {
    server.use(
      http.get('http://localhost:8000/chat/history', () =>
        HttpResponse.json([])
      )
    );
    const result = await fetchChatHistory();
    expect(result).toEqual([]);
  });

  it('token olmadan çağrıldığında da istek atılır (401 backend\'e düşer)', async () => {
    localStorage.removeItem('access_token');
    // MSW hâlâ 200 döndürüyor — interceptor olmayan bir senaryo gibi
    const result = await fetchChatHistory();
    expect(Array.isArray(result)).toBe(true);
  });
});
