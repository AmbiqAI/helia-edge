import { test, expect } from '@playwright/test';

test('public import search and inherited converter contracts', async ({page}) => {
  await page.goto('reference/');
  const browser=page.getByRole('region',{name:'Search Python API'});
  await expect(browser).toHaveAttribute('data-ready','true');
  await page.getByLabel('Import surface',{exact:true}).selectOption('Package export');
  await page.getByRole('searchbox',{name:'Search Python API'}).fill('helia_edge.metrics.ConfusionMatrix');
  await expect(browser.getByRole('link',{name:'ConfusionMatrix',exact:true})).toBeVisible();
  await page.goto('reference/api/helia_edge/converters/litert/converter/');
  await page.getByRole('link',{name:'export_header()',exact:true}).click();
  await expect(page).toHaveURL(/#helia_edge.converters.tflite.converter.TfLiteKerasConverter.export_header$/);
  await expect(page.locator('[id="helia_edge.converters.tflite.converter.TfLiteKerasConverter.export_header"]')).toBeVisible();
  await page.screenshot({path:'test-results/inherited-api.png'});
});
