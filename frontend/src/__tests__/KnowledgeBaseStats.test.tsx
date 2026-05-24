// src/__tests__/KnowledgeBaseStats.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { KnowledgeBaseStats } from '../components/KnowledgeBaseStats';
import { server } from './mocks/server';
import { statsErrorHandler } from './mocks/handlers';

// =============================================================================
// Yükleme durumu
// =============================================================================

describe('KnowledgeBaseStats — yükleme', () => {
  it('başlangıçta loading spinner gösterilir', () => {
    render(<KnowledgeBaseStats />);
    expect(screen.getByText('Yükleniyor…')).toBeInTheDocument();
  });

  it('fetch tamamlandığında spinner kaybolur', async () => {
    render(<KnowledgeBaseStats />);
    await waitFor(() =>
      expect(screen.queryByText('Yükleniyor…')).not.toBeInTheDocument()
    );
  });
});

// =============================================================================
// Başarılı veri gösterimi (MSW mock: document_count=42, topic_count=17)
// =============================================================================

describe('KnowledgeBaseStats — veri gösterimi', () => {
  it('section başlığını gösterir', async () => {
    render(<KnowledgeBaseStats />);
    await waitFor(() => screen.getByText('Bilgi Tabanı'));
    expect(screen.getByText('Bilgi Tabanı')).toBeInTheDocument();
  });

  it('alt başlığı gösterir', async () => {
    render(<KnowledgeBaseStats />);
    await waitFor(() =>
      screen.getByText('Sabancı Üniversitesi resmi kaynakları')
    );
    expect(
      screen.getByText('Sabancı Üniversitesi resmi kaynakları')
    ).toBeInTheDocument();
  });

  it('document_count değerini gösterir', async () => {
    render(<KnowledgeBaseStats />);
    // 42 < 1000 → "42" (+ eki yok)
    await waitFor(() => screen.getByText('42'));
    expect(screen.getByText('42')).toBeInTheDocument();
  });

  it('topic_count değerini gösterir', async () => {
    render(<KnowledgeBaseStats />);
    await waitFor(() => screen.getByText('17'));
    expect(screen.getByText('17')).toBeInTheDocument();
  });

  it('"Kaynak doküman" etiketini gösterir', async () => {
    render(<KnowledgeBaseStats />);
    await waitFor(() => screen.getByText('Kaynak doküman'));
    expect(screen.getByText('Kaynak doküman')).toBeInTheDocument();
  });

  it('"Kapsanan konu" etiketini gösterir', async () => {
    render(<KnowledgeBaseStats />);
    await waitFor(() => screen.getByText('Kapsanan konu'));
    expect(screen.getByText('Kapsanan konu')).toBeInTheDocument();
  });
});

// =============================================================================
// fmt() — sayı formatlama
// =============================================================================

describe('KnowledgeBaseStats — sayı formatlama', () => {
  it('1000 üzeri sayılar + eki alır', async () => {
    // MSW handler'ını override et: 4232 dönsün
    server.use(
      http.get('http://localhost/api/stats', () =>
        HttpResponse.json({
          document_count: 4232,
          topic_count: 1929,
        })
      )
    );

    render(<KnowledgeBaseStats />);
    await waitFor(() => screen.getByText('4.232+'));
    expect(screen.getByText('4.232+')).toBeInTheDocument();
    expect(screen.getByText('1.929+')).toBeInTheDocument();
  });

  it('999 altı sayılar + eki almaz', async () => {
    render(<KnowledgeBaseStats />);
    // Default mock: 42 ve 17
    await waitFor(() => screen.getByText('42'));
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.queryByText('42+')).not.toBeInTheDocument();
  });
});

// =============================================================================
// Hata durumu
// =============================================================================

describe('KnowledgeBaseStats — hata durumu', () => {
  it('fetch başarısız olunca hata mesajı gösterilir', async () => {
    server.use(statsErrorHandler);

    render(<KnowledgeBaseStats />);
    await waitFor(() =>
      screen.getByText('Veriler yüklenemedi.')
    );
    expect(screen.getByText('Veriler yüklenemedi.')).toBeInTheDocument();
  });

  it('hata durumunda spinner gösterilmez', async () => {
    server.use(statsErrorHandler);

    render(<KnowledgeBaseStats />);
    await waitFor(() =>
      expect(screen.queryByText('Yükleniyor…')).not.toBeInTheDocument()
    );
  });
});
