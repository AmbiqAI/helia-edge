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

test('horizontal tabs preserve vertical keys and reveal mobile selections', async ({page}) => {
 await page.setViewportSize({width:390,height:900});
 await page.goto('');
 await page.getByRole('button',{name:/I have a model/}).click();
 const selected=page.getByRole('tab',{name:/Evaluate/});
 await expect(selected).toHaveAttribute('aria-selected','true');
 const visible=await selected.evaluate(tab=>{
  const bounds=tab.parentElement!.getBoundingClientRect();
  const box=tab.getBoundingClientRect();
  return box.left>=bounds.left && box.right<=bounds.right;
 });
 expect(visible).toBe(true);
 const consumed=await selected.evaluate(tab=>!tab.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowDown',bubbles:true,cancelable:true})));
 expect(consumed).toBe(false);
 await expect(selected).toHaveAttribute('aria-selected','true');
 await selected.press('ArrowRight');
 await expect(page.getByRole('tab',{name:/Export/})).toHaveAttribute('aria-selected','true');
});
