// src/__tests__/LoginPage.test.tsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LoginPage } from '../components/LoginPage';
import { server } from './mocks/server';
import {
  loginUnauthorizedHandler,
  loginNetworkErrorHandler,
  adminLoginHandler,
} from './mocks/handlers';

// localStorage'ı her testten önce temizle
beforeEach(() => {
  localStorage.clear();
});

// =============================================================================
// Render
// =============================================================================

describe('LoginPage — render', () => {
  it('email input render edilir', () => {
    render(<LoginPage onLogin={vi.fn()} />);
    expect(screen.getByPlaceholderText('student@sabanciuniv.edu')).toBeInTheDocument();
  });

  it('password input render edilir', () => {
    render(<LoginPage onLogin={vi.fn()} />);
    expect(screen.getByPlaceholderText('••••••••')).toBeInTheDocument();
  });

  it('"Sign In" butonu render edilir', () => {
    render(<LoginPage onLogin={vi.fn()} />);
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  it('"Sign up" toggle butonu render edilir', () => {
    render(<LoginPage onLogin={vi.fn()} />);
    expect(screen.getByRole('button', { name: /sign up/i })).toBeInTheDocument();
  });
});

// =============================================================================
// Form validasyonu (API'ye gitmeden önce)
// =============================================================================

describe('LoginPage — form validasyonu', () => {
  it('email ve şifre boşken hata gösterir', async () => {
    render(<LoginPage onLogin={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));
    await waitFor(() =>
      expect(screen.getByText('Please fill in all fields')).toBeInTheDocument()
    );
  });

  it('sadece email girilip şifre boşken hata gösterir', async () => {
    const user = userEvent.setup();
    render(<LoginPage onLogin={vi.fn()} />);
    await user.type(screen.getByPlaceholderText('student@sabanciuniv.edu'), 'a@b.com');
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));
    await waitFor(() =>
      expect(screen.getByText('Please fill in all fields')).toBeInTheDocument()
    );
  });

  it('geçersiz email formatı hata gösterir', async () => {
    const user = userEvent.setup();
    render(<LoginPage onLogin={vi.fn()} />);
    await user.type(screen.getByPlaceholderText('student@sabanciuniv.edu'), 'gecersizemail');
    await user.type(screen.getByPlaceholderText('••••••••'), 'password123');
    fireEvent.submit(document.querySelector('form') as HTMLFormElement);
    await waitFor(() =>
      expect(screen.getByText('Please enter a valid email address')).toBeInTheDocument()
    );
  });

  it('validasyon hatası olunca API çağrısı yapılmaz', async () => {
    const onLogin = vi.fn();
    render(<LoginPage onLogin={onLogin} />);
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));
    await waitFor(() => screen.getByText('Please fill in all fields'));
    expect(onLogin).not.toHaveBeenCalled();
  });
});

// =============================================================================
// Sign up toggle
// =============================================================================

describe('LoginPage — sign up modu', () => {
  it('"Don\'t have an account?" butonuna tıklayınca name alanı çıkar', async () => {
    render(<LoginPage onLogin={vi.fn()} />);
    const toggleBtn = screen.getByRole('button', { name: /sign up/i });
    fireEvent.click(toggleBtn);
    await waitFor(() =>
      expect(screen.getByPlaceholderText('John Doe')).toBeInTheDocument()
    );
  });

  it('sign up modunda isim boşken hata gösterir', async () => {
    const user = userEvent.setup();
    render(<LoginPage onLogin={vi.fn()} />);

    // Sign up moduna geç
    fireEvent.click(screen.getByRole('button', { name: /sign up/i }));
    await waitFor(() => screen.getByPlaceholderText('John Doe'));

    // Email ve şifre doldur, isim bırak
    await user.type(screen.getByPlaceholderText('student@sabanciuniv.edu'), 'a@b.com');
    await user.type(screen.getByPlaceholderText('••••••••'), 'password123');
    fireEvent.click(screen.getByRole('button', { name: /create account/i }));

    await waitFor(() =>
      expect(screen.getByText('Please enter your name')).toBeInTheDocument()
    );
  });

  it('tekrar tıklayınca login moduna döner', async () => {
    render(<LoginPage onLogin={vi.fn()} />);
    const toggleBtn = screen.getByRole('button', { name: /sign up/i });
    fireEvent.click(toggleBtn);
    await waitFor(() => screen.getByPlaceholderText('John Doe'));

    // "Already have an account?" butonuna tıkla
    fireEvent.click(screen.getByRole('button', { name: /already have an account/i }));
    await waitFor(() =>
      expect(screen.queryByPlaceholderText('John Doe')).not.toBeInTheDocument()
    );
  });
});

// =============================================================================
// Başarılı giriş
// =============================================================================

describe('LoginPage — başarılı giriş', () => {
  it('onLogin callback çağrılır', async () => {
    const user = userEvent.setup();
    const onLogin = vi.fn();
    render(<LoginPage onLogin={onLogin} />);

    await user.type(screen.getByPlaceholderText('student@sabanciuniv.edu'), 'student@sabanciuniv.edu');
    await user.type(screen.getByPlaceholderText('••••••••'), 'password123');
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => expect(onLogin).toHaveBeenCalledOnce());
    expect(onLogin).toHaveBeenCalledWith(
      'student@sabanciuniv.edu',
      'Test Öğrencisi',
      false
    );
  });

  it('access_token localStorage\'a kaydedilir', async () => {
    const user = userEvent.setup();
    render(<LoginPage onLogin={vi.fn()} />);

    await user.type(screen.getByPlaceholderText('student@sabanciuniv.edu'), 'a@b.com');
    await user.type(screen.getByPlaceholderText('••••••••'), 'password123');
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() =>
      expect(localStorage.getItem('access_token')).toBe('mock-jwt-token-for-testing')
    );
  });

  it('admin girişinde is_admin=true iletilir', async () => {
    server.use(adminLoginHandler);
    const user = userEvent.setup();
    const onLogin = vi.fn();
    render(<LoginPage onLogin={onLogin} />);

    await user.type(screen.getByPlaceholderText('student@sabanciuniv.edu'), 'admin@sabanciuniv.edu');
    await user.type(screen.getByPlaceholderText('••••••••'), 'adminpass');
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => expect(onLogin).toHaveBeenCalledOnce());
    expect(onLogin).toHaveBeenCalledWith(
      'admin@sabanciuniv.edu',
      'Test Admin',
      true
    );
  });
});

// =============================================================================
// Hata durumları
// =============================================================================

describe('LoginPage — hata durumları', () => {
  it('401 cevabında hata mesajı gösterilir', async () => {
    server.use(loginUnauthorizedHandler);
    const user = userEvent.setup();
    render(<LoginPage onLogin={vi.fn()} />);

    await user.type(screen.getByPlaceholderText('student@sabanciuniv.edu'), 'a@b.com');
    await user.type(screen.getByPlaceholderText('••••••••'), 'yanlisSifre');
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() =>
      expect(screen.getByText('E-posta veya şifre hatalı')).toBeInTheDocument()
    );
  });

  it('401 durumunda onLogin çağrılmaz', async () => {
    server.use(loginUnauthorizedHandler);
    const user = userEvent.setup();
    const onLogin = vi.fn();
    render(<LoginPage onLogin={onLogin} />);

    await user.type(screen.getByPlaceholderText('student@sabanciuniv.edu'), 'a@b.com');
    await user.type(screen.getByPlaceholderText('••••••••'), 'yanlisSifre');
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => screen.getByText('E-posta veya şifre hatalı'));
    expect(onLogin).not.toHaveBeenCalled();
  });

  it('ağ hatasında bağlantı hatası mesajı gösterilir', async () => {
    server.use(loginNetworkErrorHandler);
    const user = userEvent.setup();
    render(<LoginPage onLogin={vi.fn()} />);

    await user.type(screen.getByPlaceholderText('student@sabanciuniv.edu'), 'a@b.com');
    await user.type(screen.getByPlaceholderText('••••••••'), 'password');
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() =>
      expect(
        screen.getByText(/Cannot connect to server|network/i)
      ).toBeInTheDocument()
    );
  });
});

// =============================================================================
// Loading state
// =============================================================================

describe('LoginPage — loading durumu', () => {
  it('submit sırasında buton "Processing..." gösterir', async () => {
    const user = userEvent.setup();
    render(<LoginPage onLogin={vi.fn()} />);

    await user.type(screen.getByPlaceholderText('student@sabanciuniv.edu'), 'a@b.com');
    await user.type(screen.getByPlaceholderText('••••••••'), 'password123');
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    // Loading başladığında "Processing..." görünür
    expect(await screen.findByText('Processing...')).toBeInTheDocument();
  });
});
