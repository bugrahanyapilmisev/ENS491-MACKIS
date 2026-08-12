// src/__tests__/SourceCard.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SourceCard, SourceReference } from '../components/SourceCard';

const baseSource: SourceReference = {
  chunk_id: 1,
  title: 'Erasmus Rehberi',
  excerpt: 'Erasmus değişim programına başvurmak için minimum GNO şartı...',
};

const sourceWithUrl: SourceReference = {
  ...baseSource,
  url: 'https://mysu.sabanciuniv.edu/erasmus',
};

// =============================================================================
// Temel render
// =============================================================================

describe('SourceCard — temel render', () => {
  it('başlığı gösterir', () => {
    render(<SourceCard source={baseSource} />);
    expect(screen.getByText('Erasmus Rehberi')).toBeInTheDocument();
  });

  it('excerpt metnini gösterir', () => {
    render(<SourceCard source={baseSource} />);
    expect(screen.getByText(/Erasmus değişim programına/)).toBeInTheDocument();
  });

  it('index numarasını başlığa ekler', () => {
    render(<SourceCard source={baseSource} index={0} />);
    expect(screen.getByText(/\[1\] Erasmus Rehberi/)).toBeInTheDocument();
  });

  it('index 1 → [2] olarak gösterilir', () => {
    render(<SourceCard source={baseSource} index={1} />);
    expect(screen.getByText(/\[2\]/)).toBeInTheDocument();
  });

  it('index verilmezse prefix olmaz', () => {
    render(<SourceCard source={baseSource} />);
    expect(screen.queryByText(/\[1\]/)).not.toBeInTheDocument();
  });
});

// =============================================================================
// URL davranışı
// =============================================================================

describe('SourceCard — URL davranışı', () => {
  it('URL varsa "Kaynağa git" bağlantı ipucu gösterilir', () => {
    render(<SourceCard source={sourceWithUrl} />);
    expect(screen.getByText('Kaynağa git')).toBeInTheDocument();
  });

  it('URL yoksa "Kaynağa git" gösterilmez', () => {
    render(<SourceCard source={baseSource} />);
    expect(screen.queryByText('Kaynağa git')).not.toBeInTheDocument();
  });

  it('URL varsa kart tıklanabilir (cursor-pointer)', () => {
    const { container } = render(<SourceCard source={sourceWithUrl} />);
    const card = container.firstChild as HTMLElement;
    expect(card.className).toContain('cursor-pointer');
  });

  it('URL yoksa kart tıklanabilir değil', () => {
    const { container } = render(<SourceCard source={baseSource} />);
    const card = container.firstChild as HTMLElement;
    expect(card.className).not.toContain('cursor-pointer');
  });

  it('URL olan karta tıklandığında window.open çağrılır', () => {
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null);
    const { container } = render(<SourceCard source={sourceWithUrl} />);
    fireEvent.click(container.firstChild as HTMLElement);
    expect(openSpy).toHaveBeenCalledWith(
      'https://mysu.sabanciuniv.edu/erasmus',
      '_blank',
      'noopener,noreferrer'
    );
    openSpy.mockRestore();
  });

  it('URL olmayan karta tıklandığında window.open çağrılmaz', () => {
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null);
    const { container } = render(<SourceCard source={baseSource} />);
    fireEvent.click(container.firstChild as HTMLElement);
    expect(openSpy).not.toHaveBeenCalled();
    openSpy.mockRestore();
  });
});

// =============================================================================
// Excerpt truncation
// =============================================================================

describe('SourceCard — excerpt', () => {
  it('150 karakterden uzun excerpt kesilmiş gösterilir', () => {
    const longExcerpt = 'A'.repeat(200);
    const source = { ...baseSource, excerpt: longExcerpt };
    render(<SourceCard source={source} />);
    // Bileşen ilk 150 karakteri alıp "..." ekliyor
    expect(screen.getByText(/A{150}\.\.\./)).toBeInTheDocument();
  });

  it('excerpt boşsa metin alanı render edilmez', () => {
    const source = { ...baseSource, excerpt: '' };
    const { container } = render(<SourceCard source={source} />);
    expect(container.querySelector('p.italic')).not.toBeInTheDocument();
  });
});
