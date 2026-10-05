import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';
import { heliaStarlight } from '@ambiqai/helia-ui/starlight';
import apiSidebar from './src/data/api-sidebar.json' with { type: 'json' };
import redirects from './src/data/redirects.json' with { type: 'json' };
import apiRedirects from './src/data/api-redirects.json' with { type: 'json' };
const base = '/helia-edge';
const page = (label, slug) => ({ label, slug });
export default defineConfig({
  site: 'https://ambiqai.github.io', base, redirects: { ...Object.fromEntries(Object.entries(redirects).map(([from,to])=>[from, `${base}${to}/`])), ...apiRedirects },
  integrations: [{name:'local-design-previews', hooks:{'astro:config:setup':({command,injectRoute})=>{
    if(command !== 'dev') return;
    injectRoute({pattern:'/mockups/',entrypoint:'./src/mockups/index.astro'});
    injectRoute({pattern:'/mockups/[variant]/',entrypoint:'./src/mockups/variant.astro'});
  }}}, react(), starlight({
    title: 'heliaEDGE', description: 'Keras components, model architectures and training tools for Edge AI.',
    favicon: '/assets/app-logo.png', customCss: ['./src/styles/site.css'],
    plugins: [heliaStarlight({
      accent: 'helia-edge', sidebar: 'always', header: { title: 'heliaEDGE', hub: { label: 'HELIA', href: 'https://ambiqai.github.io/helia-developer-hub/' } },
      sections: [
        { label: 'Home', href: `${base}/`, sidebar: false },
        { label: 'Getting started', href: `${base}/getting-started/`, sidebar: [page('Install heliaEDGE','getting-started'),page('Build your first model','getting-started/first-model'),page('Backend support','getting-started/backends')] },
        { label: 'User guide', href: `${base}/guide/`, sidebar: [
          page('Overview','guide'),
          {label:'Build models',items:[page('Choose an architecture','guide/architectures'),page('Import weights','guide/import-weights')]},
          {label:'Prepare data',items:[page('Preprocessing and augmentation','guide/preprocessing'),page('Portable signal preprocessing','guide/portable-preprocessing'),page('Input pipelines','guide/input-pipeline'),page('Preprocessing contracts','guide/preprocessing-contracts')]},
          {label:'Train and evaluate',items:[page('Training and callbacks','guide/training'),page('Masked autoencoders','guide/masked-autoencoders'),page('Metrics and evaluation','guide/evaluation')]},
          {label:'Save and deploy',items:[page('Save and load models','guide/serialization'),page('Export and quantization','guide/export')]},
          page('Development','guide/development')
        ] },
        { label: 'Examples', href: `${base}/examples/`, sidebar: [page('Overview','examples'), ...['custom-model-architecture','train-cifar-model','mlperf-tiny','miniresnet','fastenhancer','cornet','timeppg'].map((slug,i)=>page(['Custom architecture','Train CIFAR-10','MLPerf Tiny','MiniResNet','FastEnhancer','CorNET','TimePPG'][i],`examples/${slug}`))] },
        { label: 'Reference', href: `${base}/reference/`, sidebar: [page('API catalog','reference'), ...apiSidebar] },
      ],
      discoverability: { ogImage:true, jsonLd:true, markdown:true, llms:true },
      footer: { logo:'ambiq', tagline:'Part of the Ambiq HELIA AI platform', links:[{label:'GitHub',href:'https://github.com/AmbiqAI/helia-edge'},{label:'Backend support',href:`${base}/getting-started/backends/`}] },
    })],
  })],
});
