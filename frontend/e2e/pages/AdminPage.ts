// e2e/pages/AdminPage.ts
import { Page, Locator, expect } from '@playwright/test';

export class AdminPage {
  readonly page: Page;

  readonly heading: Locator;
  readonly logoutButton: Locator;
  readonly comingSoonItems: Locator;

  constructor(page: Page) {
    this.page = page;
    this.heading        = page.getByText('Admin Dashboard').first();
    this.logoutButton   = page.getByRole('button', { name: /logout/i });
    this.comingSoonItems = page.getByText(/yakında geliyor/i);
  }

  async isVisible() {
    await expect(this.heading).toBeVisible();
  }
}
