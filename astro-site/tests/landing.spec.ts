import { test, expect } from '@playwright/test';
test('guided landing connects starting points to accessible examples',async({page})=>{
 await page.goto('');
 await page.getByRole('button',{name:/I have a model/}).click();
 await expect(page.getByRole('tab',{name:/Evaluate/})).toHaveAttribute('aria-selected','true');
 await expect(page.getByRole('tabpanel')).toContainText('metric.update_state');
 await page.getByRole('tab',{name:/Evaluate/}).press('End');
 await expect(page.getByRole('tab',{name:/Export/})).toHaveAttribute('aria-selected','true');
 await page.getByRole('tab',{name:/Build/}).click();
 await expect(page.getByRole('button',{name:/I have a model/})).toHaveAttribute('aria-pressed','false');
 await page.getByRole('tabpanel').getByRole('link',{name:'Choose an architecture'}).click();
 await expect(page).toHaveURL(/guide\/architectures\/$/);
});
