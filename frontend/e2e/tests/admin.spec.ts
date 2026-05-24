// e2e/tests/admin.spec.ts
import { test, expect } from '@playwright/test';
import { LoginPage } from '../pages/LoginPage';
import { ChatPage }  from '../pages/ChatPage';
import { AdminPage } from '../pages/AdminPage';
import { mockAdminLogin, mockLoginSuccess, mockHistory, mockStats } from '../fixtures/mocks';

// ─── Admin Dashboard ─────────────────────────────────────────────────────────

test('admin girişi → AdminDashboard gösterilir', async ({ page }) => {
  await mockAdminLogin(page);
  await page.goto('/');
  const loginPage = new LoginPage(page);
  await loginPage.login('admin@sabanciuniv.edu', 'adminpass');

  const adminPage = new AdminPage(page);
  await adminPage.isVisible();
});

test('admin dashboard\'da chat textarea görünmez', async ({ page }) => {
  await mockAdminLogin(page);
  await page.goto('/');
  const loginPage = new LoginPage(page);
  await loginPage.login('admin@sabanciuniv.edu', 'adminpass');

  await expect(page.getByPlaceholder('Ask anything about the university...')).toBeHidden();
});

test('admin dashboard\'da "Coming soon" placeholder\'ları görünür', async ({ page }) => {
  await mockAdminLogin(page);
  await page.goto('/');
  const loginPage = new LoginPage(page);
  await loginPage.login('admin@sabanciuniv.edu', 'adminpass');

  const adminPage = new AdminPage(page);
  // En az bir "yakında geliyor" metni olmalı
  await expect(adminPage.comingSoonItems.first()).toBeVisible();
});

test('admin logout → login sayfasına dönüş', async ({ page }) => {
  await mockAdminLogin(page);
  await page.goto('/');
  const loginPage = new LoginPage(page);
  await loginPage.login('admin@sabanciuniv.edu', 'adminpass');

  const adminPage = new AdminPage(page);
  await adminPage.isVisible();
  await adminPage.logoutButton.click();

  // Login sayfası tekrar görünmeli
  await expect(loginPage.signInButton).toBeVisible();
});

// ─── Normal kullanıcı admin dashboard'a erişemez ─────────────────────────────

test('normal kullanıcı girişi → admin dashboard görünmez', async ({ page }) => {
  await mockLoginSuccess(page);
  await mockHistory(page, []);
  await mockStats(page);
  await page.goto('/');

  const loginPage = new LoginPage(page);
  await loginPage.login('student@sabanciuniv.edu', 'password123');

  // "Admin Dashboard" başlığı görünmemeli
  await expect(page.getByText('Admin Dashboard').first()).toBeHidden();
  // Chat arayüzü görünmeli
  await expect(page.getByPlaceholder('Ask anything about the university...')).toBeVisible();
});
