// e2e/pages/LoginPage.ts
import { Page, Locator, expect } from '@playwright/test';

export class LoginPage {
  readonly page: Page;

  // Locators
  readonly emailInput: Locator;
  readonly passwordInput: Locator;
  readonly nameInput: Locator;
  readonly signInButton: Locator;
  readonly signUpButton: Locator;
  readonly createAccountButton: Locator;
  readonly errorAlert: Locator;

  constructor(page: Page) {
    this.page = page;
    this.emailInput    = page.getByPlaceholder('student@sabanciuniv.edu');
    this.passwordInput = page.getByPlaceholder('••••••••');
    this.nameInput     = page.getByPlaceholder('John Doe');
    this.signInButton  = page.getByRole('button', { name: /sign in/i });
    // "Don't have an account? Sign up" toggle button
    this.signUpButton        = page.getByRole('button', { name: /don't have an account/i });
    this.createAccountButton = page.getByRole('button', { name: /create account/i });
    this.errorAlert   = page.locator('[role="alert"], .border-destructive\\/50').first();
  }

  async goto() {
    await this.page.goto('/');
  }

  async login(email: string, password: string) {
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
    await this.signInButton.click();
  }

  async switchToSignUp() {
    await this.signUpButton.click();
  }

  async isVisible() {
    await expect(this.signInButton).toBeVisible();
  }

  async errorMessage() {
    return this.errorAlert.textContent();
  }
}
