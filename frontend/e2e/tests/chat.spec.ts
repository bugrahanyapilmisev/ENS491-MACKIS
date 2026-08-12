// e2e/tests/chat.spec.ts
import { test, expect } from '@playwright/test';
import { LoginPage } from '../pages/LoginPage';
import { ChatPage }  from '../pages/ChatPage';
import {
  mockLoginSuccess,
  mockHistory,
  mockChat,
  mockStats,
  chatResponse,
} from '../fixtures/mocks';

// Her testten önce başarılı giriş yap ve chat sayfasına geç
async function setupChat(page: any, historyData = []) {
  await mockLoginSuccess(page);
  await mockHistory(page, historyData);
  await mockChat(page);
  await mockStats(page);

  await page.goto('/');
  const loginPage = new LoginPage(page);
  await loginPage.login('student@sabanciuniv.edu', 'password123');

  const chatPage = new ChatPage(page);
  await chatPage.isVisible();
  return chatPage;
}

// ─── Welcome Screen ──────────────────────────────────────────────────────────

test('giriş sonrası welcome ekranı görünür', async ({ page }) => {
  const chatPage = await setupChat(page);
  await expect(chatPage.welcomeScreen).toBeVisible();
});

test('welcome ekranında örnek soru butonları görünür', async ({ page }) => {
  const chatPage = await setupChat(page);
  // En az 4 örnek soru butonu olmalı
  const count = await chatPage.sampleQuestionButtons.count();
  expect(count).toBeGreaterThanOrEqual(4);
});

// ─── Mesaj Gönderme ──────────────────────────────────────────────────────────

test('mesaj yazıp Enter → kullanıcı mesajı ekranda görünür', async ({ page }) => {
  const chatPage = await setupChat(page);
  const question = 'Erasmus GNO kaç olmalı?';

  await chatPage.sendMessage(question);
  // Use .whitespace-pre-wrap to target only the chat message bubble, not sidebar preview or sample buttons
  await expect(page.locator('.whitespace-pre-wrap', { hasText: question })).toBeVisible();
});

test('mesaj gönderilince typing indicator görünür', async ({ page }) => {
  // Mock'u geciktirerek typing indicator'ü yakalayabilelim
  await mockLoginSuccess(page);
  await mockHistory(page, []);
  await mockStats(page);
  await page.route('**/chat', async (route) => {
    // 500ms gecikme ekle
    await new Promise((r) => setTimeout(r, 500));
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(chatResponse),
    });
  });

  await page.goto('/');
  const loginPage = new LoginPage(page);
  await loginPage.login('student@sabanciuniv.edu', 'password123');
  const chatPage = new ChatPage(page);
  await chatPage.isVisible();

  await chatPage.sendMessage('Test sorusu');
  await expect(chatPage.typingIndicator).toBeVisible();
});

test('cevap gelince MACKIS mesajı görünür', async ({ page }) => {
  const chatPage = await setupChat(page);

  await chatPage.sendMessage('GNO sorusu');
  await chatPage.waitForAssistantReply();

  await expect(page.getByText(chatResponse.answer)).toBeVisible();
});

test('confidence badge cevapla birlikte görünür', async ({ page }) => {
  const chatPage = await setupChat(page);

  await chatPage.sendMessage('GNO sorusu');
  await chatPage.waitForAssistantReply();

  // confidence 0.92 → "92% confidence"
  await expect(page.getByText('92% confidence')).toBeVisible();
});

test('sources collapsible görünür ve açılır', async ({ page }) => {
  const chatPage = await setupChat(page);

  await chatPage.sendMessage('GNO sorusu');
  await chatPage.waitForAssistantReply();

  // Collapsible trigger
  const trigger = page.getByText(/Referenced 1 document/);
  await expect(trigger).toBeVisible();
  await trigger.click();

  // Kaynak kartı açılmalı
  await expect(page.getByText(/Erasmus Rehberi/)).toBeVisible();
});

// ─── Örnek Soru Tıklama ──────────────────────────────────────────────────────

test('örnek soruya tıklama → mesaj gönderilir ve cevap gelir', async ({ page }) => {
  const chatPage = await setupChat(page);

  // Welcome ekranındaki ilk örnek soru butonuna tıkla
  const firstSample = chatPage.sampleQuestionButtons.first();
  await firstSample.click();

  // textContent() returns emoji+title+subtitle concatenated — check the chat bubble directly instead
  // Ensure a user message appeared in the main chat area
  await expect(page.locator('.whitespace-pre-wrap').first()).toBeVisible();
  // Cevap gelmeli
  await chatPage.waitForAssistantReply();
  await expect(page.getByText(chatResponse.answer)).toBeVisible();
});

// ─── Yeni Konuşma ────────────────────────────────────────────────────────────

test('"New Conversation" butonu → welcome ekranına döner', async ({ page }) => {
  // Önce bir mesaj gönder
  const chatPage = await setupChat(page);
  await chatPage.sendMessage('Bir soru');
  await chatPage.waitForAssistantReply();

  // Yeni konuşma
  await chatPage.newConversationButton.click();
  await expect(chatPage.welcomeScreen).toBeVisible();
});

// ─── Model Seçici ────────────────────────────────────────────────────────────

test('model seçici görünür ve seçim yapılabilir', async ({ page }) => {
  const chatPage = await setupChat(page);
  await expect(chatPage.modelSelect).toBeVisible();

  const options = await chatPage.modelSelect.locator('option').count();
  expect(options).toBeGreaterThan(1);
});
