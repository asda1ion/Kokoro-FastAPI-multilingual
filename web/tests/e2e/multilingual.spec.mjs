import { expect, test } from './fixtures/app.mjs';

test('Multilingual is selected by default and sends multi for mixed text', async ({ page }) => {
    await page.route('**/v1/audio/speech', (route) => route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({ detail: { message: 'Request captured' } }),
    }));
    await page.goto('/');
    await expect(page.locator('#lang-select')).toHaveValue('multi');
    await expect(page.locator('#lang-select option:checked')).toHaveText('Multilingual');
    await expect(page.locator('.selected-voice-tag').first()).toBeVisible();
    await page.locator('.page-content').fill('Привет, people!');
    const request = page.waitForRequest('**/v1/audio/speech');
    await page.locator('#generate-btn').click();
    expect((await request).postDataJSON()).toMatchObject({
        input: 'Привет, people!',
        lang_code: 'multi',
    });
});
