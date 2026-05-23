// e2e/pages/ChatPage.ts
import { Page, Locator, expect } from '@playwright/test';

export class ChatPage {
  readonly page: Page;

  // Header
  readonly header: Locator;
  readonly logoutButton: Locator;
  readonly mackisBadge: Locator;

  // Sidebar
  readonly newConversationButton: Locator;
  readonly conversationItems: Locator;

  // Chat area
  readonly welcomeScreen: Locator;
  readonly sampleQuestionButtons: Locator;
  readonly messageList: Locator;
  readonly typingIndicator: Locator;

  // Input
  readonly chatTextarea: Locator;
  readonly sendButton: Locator;
  readonly modelSelect: Locator;

  constructor(page: Page) {
    this.page = page;

    this.header             = page.locator('header');
    this.logoutButton       = page.getByRole('button', { name: /logout/i });
    this.mackisBadge        = page.getByText('RAG-Powered');

    this.newConversationButton = page.getByRole('button', { name: /new conversation/i });
    this.conversationItems     = page.locator('[data-sidebar="menu-button"]');

    this.welcomeScreen         = page.getByText("MACKIS'e Hoş Geldiniz");
    this.sampleQuestionButtons = page.locator('button').filter({ hasText: /GNO|Erasmus|Kütüphane|Burs|Mezuniyet|Çift anadal|Uyarı/i });

    this.messageList     = page.locator('.divide-y > div');
    this.typingIndicator = page.getByText('Searching university knowledge base...');

    this.chatTextarea = page.getByPlaceholder('Ask anything about the university...');
    this.sendButton   = page.locator('button[type="submit"]').last();
    this.modelSelect  = page.locator('select');
  }

  /** Belirli içeriğe sahip kullanıcı/asistan mesajını bulur */
  messageByText(text: string) {
    return this.page.locator('div').filter({ hasText: text }).last();
  }

  /** "You" etiketli mesaj bloklarını döner */
  get userMessages() {
    return this.page.locator('div').filter({ hasText: /^You$/ });
  }

  /** "MACKIS" etiketli mesaj bloklarını döner */
  get assistantMessages() {
    return this.page.locator('div').filter({ hasText: /^MACKIS$/ });
  }

  async sendMessage(text: string) {
    await this.chatTextarea.fill(text);
    await this.chatTextarea.press('Enter');
  }

  async waitForAssistantReply() {
    // Typing indicator çıkar, sonra kaybolur
    await expect(this.typingIndicator).toBeVisible({ timeout: 5_000 }).catch(() => {});
    await expect(this.typingIndicator).toBeHidden({ timeout: 10_000 });
  }

  async isVisible() {
    await expect(this.chatTextarea).toBeVisible();
  }
}
