// e2e/tests/auth.spec.ts
import { test, expect } from '@playwright/test';
import { LoginPage } from '../pages/LoginPage';
import { ChatPage }  from '../pages/ChatPage';
import { AdminPage } from '../pages/AdminPage';
import {
  mockLoginSuccess,
  mockLoginUnauthorized,
  mockLoginNetworkError,
  mockAdminLogin,
  mockHistory,
  mockChat,
  mockStats,
} from '../fixtures/mocks';

// ─── Render ──────────────────────────────────────────────────────────────────

test('login sayfası doğru render edilir', async ({ page }) => {
  await page.goto('/');
  const loginPage = new LoginPage(page);

  await expect(loginPage.emailInput).toBeVisible();
  await expect(loginPage.passwordInput).toBeVisible();
  await expect(loginPage.signInButton).toBeVisible();
});

// ─── Form Validasyon ─────────────────────────────────────────────────────────

test('boş alanlarla submit → hata mesajı gösterilir', async ({ page }) => {
  await page.goto('/');
  const loginPage = new LoginPage(page);

  await loginPage.signInButton.click();
  await expect(page.getByText('Please fill in all fields')).toBeVisible();
});

test('geçersiz email formatı → hata mesajı gösterilir', async ({ page }) => {
  await page.goto('/');
  const loginPage = new LoginPage(page);

  await loginPage.emailInput.fill('gecersizemail');
  await loginPage.passwordInput.fill('password123');
  // form.submit() ile HTML5 validasyonunu bypass et
  await page.evaluate(() => {
    (document.querySelector('form') as HTMLFormElement).dispatchEvent(
      new Event('submit', { bubbles: true, cancelable: true })
    );
  });
  await expect(page.getByText('Please enter a valid email address')).toBeVisible();
});

// ─── Başarısız Giriş ─────────────────────────────────────────────────────────

test('yanlış şifre (401) → Türkçe hata mesajı gösterilir', async ({ page }) => {
  await mockLoginUnauthorized(page);
  await page.goto('/');
  const loginPage = new LoginPage(page);

  await loginPage.login('a@b.com', 'yanlisSifre');
  await expect(page.getByText('E-posta veya şifre hatalı')).toBeVisible();
});

test('sunucuya bağlanılamıyor → bağlantı hata mesajı gösterilir', async ({ page }) => {
  await mockLoginNetworkError(page);
  await page.goto('/');
  const loginPage = new LoginPage(page);

  await loginPage.login('a@b.com', 'password');
  await expect(page.getByText(/cannot connect to server/i)).toBeVisible();
});

// ─── Başarılı Giriş ──────────────────────────────────────────────────────────

test('başarılı giriş → chat arayüzüne geçiş', async ({ page }) => {
  await mockLoginSuccess(page);
  await mockHistory(page, []);   // boş geçmiş
  await mockStats(page);
  await page.goto('/');
  const loginPage = new LoginPage(page);
  const chatPage  = new ChatPage(page);

  await loginPage.login('student@sabanciuniv.edu', 'password123');
  await chatPage.isVisible();
  await expect(chatPage.welcomeScreen).toBeVisible();
});

// ─── Admin Girişi ─────────────────────────────────────────────────────────────

test('admin girişi → AdminDashboard görünür, chat görünmez', async ({ page }) => {
  await mockAdminLogin(page);
  await page.goto('/');
  const loginPage = new LoginPage(page);
  const adminPage = new AdminPage(page);

  await loginPage.login('admin@sabanciuniv.edu', 'adminpass');
  await adminPage.isVisible();
  // Chat textarea görünmemeli
  await expect(page.getByPlaceholder('Ask anything about the university...')).toBeHidden();
});

// ─── Logout ──────────────────────────────────────────────────────────────────

test('logout → login sayfasına dönüş', async ({ page }) => {
  await mockLoginSuccess(page);
  await mockHistory(page, []);
  await mockStats(page);
  await page.goto('/');
  const loginPage = new LoginPage(page);
  const chatPage  = new ChatPage(page);

  await loginPage.login('student@sabanciuniv.edu', 'password123');
  await chatPage.isVisible();

  await chatPage.logoutButton.click();
  // Login sayfası tekrar görünmeli
  await expect(loginPage.signInButton).toBeVisible();
});
