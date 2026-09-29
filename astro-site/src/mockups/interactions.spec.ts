import { test, expect } from '@playwright/test';
for(const id of ['01','02','03','04']) {
 test(`landing concept ${id}`,async({page})=>{
  await page.goto(`mockups/${id}/`);
  await expect(page.locator('h1')).toHaveCount(1);
  await page.getByRole('tab',{name:/Export/}).click();
  await expect(page.getByRole('tabpanel')).toContainText('converter.export');
  await page.getByRole('tab',{name:/Export/}).press('Home');
  await expect(page.getByRole('tab',{name:/Prepare/})).toHaveAttribute('aria-selected','true');
  if(id==='03'||id==='04'){
   await page.getByRole('button',{name:/I have a model/}).click();
   await expect(page.getByRole('tab',{name:/Evaluate/})).toHaveAttribute('aria-selected','true');
  }
  await page.getByRole('tab',{name:/Build/}).click();
  for(const width of [1280,390]){
   await page.setViewportSize({width,height:960});
   await page.evaluate(()=>scrollTo(0,0));
   expect(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth+1)).toBe(true);
   await page.screenshot({path:`test-results/concept-${id}-${width}.png`,fullPage:true});
  }
  await page.getByRole('button',{name:'Toggle color theme'}).click();
  await expect(page.locator('html')).toHaveClass('dark');
  await page.screenshot({path:`test-results/concept-${id}-dark.png`,fullPage:true});
 });
}
