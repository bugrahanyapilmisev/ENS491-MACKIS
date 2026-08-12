// src/__tests__/ChatMessage.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ChatMessage } from '../components/ChatMessage';
import { SourceReference } from '../components/SourceCard';

const mockSources: SourceReference[] = [
  {
    chunk_id: 1,
    title: 'Erasmus Rehberi',
    excerpt: 'Erasmus başvurusu için minimum GNO şartı 2.5 olmalıdır.',
    url: 'https://mysu.sabanciuniv.edu/erasmus',
  },
  {
    chunk_id: 2,
    title: 'Öğrenci El Kitabı',
    excerpt: 'Burs başvuruları her yıl Ekim ayında açılmaktadır.',
  },
];

// =============================================================================
// Kullanıcı mesajı
// =============================================================================

describe('ChatMessage — kullanıcı mesajı', () => {
  it('"You" etiketi gösterilir', () => {
    render(<ChatMessage role="user" content="Merhaba" />);
    expect(screen.getByText('You')).toBeInTheDocument();
  });

  it('mesaj içeriği gösterilir', () => {
    render(<ChatMessage role="user" content="Erasmus için GNO kaç olmalı?" />);
    expect(screen.getByText('Erasmus için GNO kaç olmalı?')).toBeInTheDocument();
  });

  it('sources collapsible gösterilmez', () => {
    render(<ChatMessage role="user" content="test" sources={mockSources} />);
    expect(screen.queryByText(/Referenced/)).not.toBeInTheDocument();
  });

  it('confidence badge gösterilmez', () => {
    render(<ChatMessage role="user" content="test" confidence={0.95} />);
    expect(screen.queryByText(/confidence/)).not.toBeInTheDocument();
  });
});

// =============================================================================
// Asistan mesajı
// =============================================================================

describe('ChatMessage — asistan mesajı', () => {
  it('"MACKIS" etiketi gösterilir', () => {
    render(<ChatMessage role="assistant" content="Merhaba, size yardımcı olabilirim." />);
    expect(screen.getByText('MACKIS')).toBeInTheDocument();
  });

  it('"You" etiketi gösterilmez', () => {
    render(<ChatMessage role="assistant" content="test" />);
    expect(screen.queryByText('You')).not.toBeInTheDocument();
  });

  it('markdown bold metni render edilir', () => {
    render(<ChatMessage role="assistant" content="Bu **önemli** bir bilgi." />);
    const bold = screen.getByText('önemli');
    expect(bold.tagName).toBe('STRONG');
  });

  it('markdown link render edilir ve target=_blank olur', () => {
    render(
      <ChatMessage
        role="assistant"
        content="[Sabancı](https://sabanciuniv.edu) üniversitesi"
      />
    );
    const link = screen.getByRole('link', { name: 'Sabancı' });
    expect(link).toHaveAttribute('href', 'https://sabanciuniv.edu');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer');
  });

  it('timestamp gösterilir', () => {
    render(<ChatMessage role="assistant" content="test" timestamp="10:30" />);
    expect(screen.getByText('10:30')).toBeInTheDocument();
  });
});

// =============================================================================
// Confidence badge
// =============================================================================

describe('ChatMessage — confidence badge', () => {
  it('≥90% → badge gösterilir', () => {
    render(<ChatMessage role="assistant" content="test" confidence={0.95} />);
    expect(screen.getByText('95% confidence')).toBeInTheDocument();
  });

  it('75–89% arası → badge gösterilir', () => {
    render(<ChatMessage role="assistant" content="test" confidence={0.80} />);
    expect(screen.getByText('80% confidence')).toBeInTheDocument();
  });

  it('<75% → badge gösterilir (sarı)', () => {
    render(<ChatMessage role="assistant" content="test" confidence={0.60} />);
    expect(screen.getByText('60% confidence')).toBeInTheDocument();
  });

  it('confidence=0 → badge gösterilmez', () => {
    render(<ChatMessage role="assistant" content="test" confidence={0} />);
    expect(screen.queryByText(/confidence/)).not.toBeInTheDocument();
  });

  it('confidence verilmezse badge gösterilmez', () => {
    render(<ChatMessage role="assistant" content="test" />);
    expect(screen.queryByText(/confidence/)).not.toBeInTheDocument();
  });
});

// =============================================================================
// Sources collapsible
// =============================================================================

describe('ChatMessage — sources collapsible', () => {
  it('sources boşsa collapsible gösterilmez', () => {
    render(<ChatMessage role="assistant" content="test" sources={[]} />);
    expect(screen.queryByText(/Referenced/)).not.toBeInTheDocument();
  });

  it('sources varsa kaç doküman referanslandığı gösterilir', () => {
    render(<ChatMessage role="assistant" content="test" sources={mockSources} />);
    expect(screen.getByText(/Referenced 2 documents/)).toBeInTheDocument();
  });

  it('tek source için "document" (tekil) kullanılır', () => {
    render(<ChatMessage role="assistant" content="test" sources={[mockSources[0]]} />);
    expect(
      screen.getByText((_, el) => !!el?.textContent?.match(/Referenced 1 document$/))
    ).toBeInTheDocument();
  });

  it('başlangıçta source kartları gizlidir', () => {
    render(<ChatMessage role="assistant" content="test" sources={mockSources} />);
    expect(screen.queryByText('Erasmus Rehberi')).not.toBeInTheDocument();
  });

  it('trigger tıklanınca source kartları açılır', () => {
    render(<ChatMessage role="assistant" content="test" sources={mockSources} />);
    const trigger = screen.getByText(/Referenced 2 documents/).closest('button') ||
                    screen.getByText(/Referenced 2 documents/);
    fireEvent.click(trigger);
    expect(screen.getByText(/Erasmus Rehberi/)).toBeInTheDocument();
    expect(screen.getByText(/Öğrenci El Kitabı/)).toBeInTheDocument();
  });

  it('tekrar tıklanınca source kartları kapanır', () => {
    render(<ChatMessage role="assistant" content="test" sources={mockSources} />);
    const trigger = screen.getByText(/Referenced 2 documents/);
    fireEvent.click(trigger);  // aç
    fireEvent.click(trigger);  // kapat
    expect(screen.queryByText(/Erasmus Rehberi/)).not.toBeInTheDocument();
  });
});
